//===--- AvoidCStyleCastsCheck.cpp - clang-tidy -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AvoidCStyleCastsCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace google {
namespace readability {

void
AvoidCStyleCastsCheck::registerMatchers(ast_matchers::MatchFinder *Finder) {
  Finder->addMatcher(
      cStyleCastExpr(
          // Filter out (EnumType)IntegerLiteral construct, which is generated
          // for non-type template arguments of enum types.
          // FIXME: Remove this once this is fixed in the AST.
          unless(hasParent(substNonTypeTemplateParmExpr())),
          // Avoid matches in template instantiations.
          unless(isInTemplateInstantiation())).bind("cast"),
      this);
}

static bool needsConstCast(QualType SourceType, QualType DestType) {
  SourceType = SourceType.getNonReferenceType();
  DestType = DestType.getNonReferenceType();
  while (SourceType->isPointerType() && DestType->isPointerType()) {
    SourceType = SourceType->getPointeeType();
    DestType = DestType->getPointeeType();
    if (SourceType.isConstQualified() && !DestType.isConstQualified())
      return true;
  }
  return false;
}

static bool pointedTypesAreEqual(QualType SourceType, QualType DestType) {
  SourceType = SourceType.getNonReferenceType();
  DestType = DestType.getNonReferenceType();
  while (SourceType->isPointerType() && DestType->isPointerType()) {
    SourceType = SourceType->getPointeeType();
    DestType = DestType->getPointeeType();
  }
  return SourceType.getUnqualifiedType() == DestType.getUnqualifiedType();
}

void AvoidCStyleCastsCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *CastExpr = Result.Nodes.getNodeAs<CStyleCastExpr>("cast");

  auto ParenRange = CharSourceRange::getTokenRange(CastExpr->getLParenLoc(),
                                                   CastExpr->getRParenLoc());
  // Ignore casts in macros.
  if (ParenRange.getBegin().isMacroID() || ParenRange.getEnd().isMacroID())
    return;

  // Casting to void is an idiomatic way to mute "unused variable" and similar
  // warnings.
  if (CastExpr->getTypeAsWritten()->isVoidType())
    return;

  QualType SourceType = CastExpr->getSubExprAsWritten()->getType();
  QualType DestType = CastExpr->getTypeAsWritten();

  if (SourceType == DestType) {
    diag(CastExpr->getLocStart(), "redundant cast to the same type")
        << FixItHint::CreateRemoval(ParenRange);
    return;
  }
  SourceType = SourceType.getCanonicalType();
  DestType = DestType.getCanonicalType();
  if (SourceType == DestType) {
    diag(CastExpr->getLocStart(),
         "possibly redundant cast between typedefs of the same type");
    return;
  }


  // The rest of this check is only relevant to C++.
  if (!Result.Context->getLangOpts().CPlusPlus)
    return;
  // Ignore code inside extern "C" {} blocks.
  if (!match(expr(hasAncestor(linkageSpecDecl())), *CastExpr, *Result.Context)
           .empty())
    return;
  // Ignore code in .c files and headers included from them, even if they are
  // compiled as C++.
  if (getCurrentMainFile().endswith(".c"))
    return;
  // Ignore code in .c files #included in other files (which shouldn't be done,
  // but people still do this for test and other purposes).
  SourceManager &SM = *Result.SourceManager;
  if (SM.getFilename(SM.getSpellingLoc(CastExpr->getLocStart())).endswith(".c"))
    return;

  // Leave type spelling exactly as it was (unlike
  // getTypeAsWritten().getAsString() which would spell enum types 'enum X').
  StringRef DestTypeString =
      Lexer::getSourceText(CharSourceRange::getTokenRange(
                               CastExpr->getLParenLoc().getLocWithOffset(1),
                               CastExpr->getRParenLoc().getLocWithOffset(-1)),
                           SM, Result.Context->getLangOpts());

  auto diag_builder =
      diag(CastExpr->getLocStart(), "C-style casts are discouraged; use %0");

  auto ReplaceWithCast = [&](StringRef CastType) {
    diag_builder << CastType;

    const Expr *SubExpr = CastExpr->getSubExprAsWritten()->IgnoreImpCasts();
    std::string CastText = (CastType + "<" + DestTypeString + ">").str();
    if (!isa<ParenExpr>(SubExpr)) {
      CastText.push_back('(');
      diag_builder << FixItHint::CreateInsertion(
          Lexer::getLocForEndOfToken(SubExpr->getLocEnd(), 0, SM,
                                     Result.Context->getLangOpts()),
          ")");
    }
    diag_builder << FixItHint::CreateReplacement(ParenRange, CastText);
  };

  // Suggest appropriate C++ cast. See [expr.cast] for cast notation semantics.
  switch (CastExpr->getCastKind()) {
  case CK_NoOp:
    if (needsConstCast(SourceType, DestType) &&
        pointedTypesAreEqual(SourceType, DestType)) {
      ReplaceWithCast("const_cast");
      return;
    }
    if (DestType->isReferenceType() &&
        (SourceType.getNonReferenceType() ==
             DestType.getNonReferenceType().withConst() ||
         SourceType.getNonReferenceType() == DestType.getNonReferenceType())) {
      ReplaceWithCast("const_cast");
      return;
    }
    // FALLTHROUGH
  case clang::CK_IntegralCast:
    // Convert integral and no-op casts between builtin types and enums to
    // static_cast. A cast from enum to integer may be unnecessary, but it's
    // still retained.
    if ((SourceType->isBuiltinType() || SourceType->isEnumeralType()) &&
        (DestType->isBuiltinType() || DestType->isEnumeralType())) {
      ReplaceWithCast("static_cast");
      return;
    }
    break;
  case CK_BitCast:
    // FIXME: Suggest const_cast<...>(reinterpret_cast<...>(...)) replacement.
    if (!needsConstCast(SourceType, DestType)) {
      ReplaceWithCast("reinterpret_cast");
      return;
    }
    break;
  default:
    break;
  }

  diag_builder << "static_cast/const_cast/reinterpret_cast";
}

} // namespace readability
} // namespace google
} // namespace tidy
} // namespace clang
