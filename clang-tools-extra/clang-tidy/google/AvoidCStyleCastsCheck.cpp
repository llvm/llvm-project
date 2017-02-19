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

void AvoidCStyleCastsCheck::registerMatchers(
    ast_matchers::MatchFinder *Finder) {
  Finder->addMatcher(
      cStyleCastExpr(
          // Filter out (EnumType)IntegerLiteral construct, which is generated
          // for non-type template arguments of enum types.
          // FIXME: Remove this once this is fixed in the AST.
          unless(hasParent(substNonTypeTemplateParmExpr())),
          // Avoid matches in template instantiations.
          unless(isInTemplateInstantiation()))
          .bind("cast"),
      this);
}

static bool needsConstCast(QualType SourceType, QualType DestType) {
  for (;;) {
    if (SourceType.isConstQualified() && !DestType.isConstQualified()) {
      return (SourceType->isPointerType() == DestType->isPointerType()) &&
             (SourceType->isReferenceType() == DestType->isReferenceType());
    }
    if ((SourceType->isPointerType() && DestType->isPointerType()) ||
        (SourceType->isReferenceType() && DestType->isReferenceType())) {
      SourceType = SourceType->getPointeeType();
      DestType = DestType->getPointeeType();
   } else {
     return false;
   }
  }
}

static bool pointedUnqualifiedTypesAreEqual(QualType T1, QualType T2) {
  while ((T1->isPointerType() && T2->isPointerType()) ||
         (T1->isReferenceType() && T2->isReferenceType())) {
    T1 = T1->getPointeeType();
    T2 = T2->getPointeeType();
  }
  return T1.getUnqualifiedType() == T2.getUnqualifiedType();
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
  if (CastExpr->getCastKind() == CK_ToVoid)
    return;

  auto isFunction = [](QualType T) {
    T = T.getCanonicalType().getNonReferenceType();
    return T->isFunctionType() || T->isFunctionPointerType() ||
           T->isMemberFunctionPointerType();
  };

  const QualType DestTypeAsWritten = CastExpr->getTypeAsWritten();
  const QualType SourceTypeAsWritten = CastExpr->getSubExprAsWritten()->getType();
  const QualType SourceType = SourceTypeAsWritten.getCanonicalType();
  const QualType DestType = DestTypeAsWritten.getCanonicalType();

  bool FnToFnCast =
      isFunction(SourceTypeAsWritten) && isFunction(DestTypeAsWritten);

  if (CastExpr->getCastKind() == CK_NoOp && !FnToFnCast) {
    // Function pointer/reference casts may be needed to resolve ambiguities in
    // case of overloaded functions, so detection of redundant casts is trickier
    // in this case. Don't emit "redundant cast" warnings for function
    // pointer/reference types.
    if (SourceTypeAsWritten == DestTypeAsWritten) {
      diag(CastExpr->getLocStart(), "redundant cast to the same type")
          << FixItHint::CreateRemoval(ParenRange);
      return;
    }
    if (SourceType == DestType) {
      diag(CastExpr->getLocStart(),
           "possibly redundant cast between typedefs of the same type");
      return;
    }
  }

  // The rest of this check is only relevant to C++.
  if (!getLangOpts().CPlusPlus)
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
                           SM, getLangOpts());

  auto Diag =
      diag(CastExpr->getLocStart(), "C-style casts are discouraged; use %0");

  auto ReplaceWithCast = [&](std::string CastText) {
    const Expr *SubExpr = CastExpr->getSubExprAsWritten()->IgnoreImpCasts();
    if (!isa<ParenExpr>(SubExpr)) {
      CastText.push_back('(');
      Diag << FixItHint::CreateInsertion(
          Lexer::getLocForEndOfToken(SubExpr->getLocEnd(), 0, SM,
                                     getLangOpts()),
          ")");
    }
    Diag << FixItHint::CreateReplacement(ParenRange, CastText);
  };
  auto ReplaceWithNamedCast = [&](StringRef CastType) {
    Diag << CastType;
    std::string CastText = (CastType + "<" + DestTypeString + ">").str();
    ReplaceWithCast(std::move(CastText));
  };
  auto ReplaceWithCtorCall = [&]() {
    std::string CastText;
    if (!DestTypeAsWritten.hasQualifiers() &&
        DestTypeAsWritten->isRecordType() &&
        !DestTypeAsWritten->isElaboratedTypeSpecifier()) {
      Diag << "constructor call syntax";
      CastText = DestTypeString.str(); // FIXME: Validate DestTypeString, maybe.
    } else {
      Diag << "static_cast";
      CastText = ("static_cast<" + DestTypeString + ">").str();
    }
    ReplaceWithCast(std::move(CastText));
  };

  // Suggest appropriate C++ cast. See [expr.cast] for cast notation semantics.
  switch (CastExpr->getCastKind()) {
  case CK_FunctionToPointerDecay:
    ReplaceWithNamedCast("static_cast");
    return;
  case CK_ConstructorConversion:
    ReplaceWithCtorCall();
    return;
  case CK_NoOp:
    if (FnToFnCast) {
      ReplaceWithNamedCast("static_cast");
      return;
    }
    if (needsConstCast(SourceType, DestType) &&
        pointedUnqualifiedTypesAreEqual(SourceType, DestType)) {
      ReplaceWithNamedCast("const_cast");
      return;
    }
    if (DestType->isReferenceType()) {
      QualType Dest = DestType.getNonReferenceType();
      QualType Source = SourceType.getNonReferenceType();
      if (Source == Dest.withConst() ||
          SourceType.getNonReferenceType() == DestType.getNonReferenceType()) {
        ReplaceWithNamedCast("const_cast");
        return;
      }
      break;
    }
  // FALLTHROUGH
  case clang::CK_IntegralCast:
    // Convert integral and no-op casts between builtin types and enums to
    // static_cast. A cast from enum to integer may be unnecessary, but it's
    // still retained.
    if ((SourceType->isBuiltinType() || SourceType->isEnumeralType()) &&
        (DestType->isBuiltinType() || DestType->isEnumeralType())) {
      ReplaceWithNamedCast("static_cast");
      return;
    }
    break;
  case CK_BitCast:
    // FIXME: Suggest const_cast<...>(reinterpret_cast<...>(...)) replacement.
    if (!needsConstCast(SourceType, DestType)) {
      ReplaceWithNamedCast("reinterpret_cast");
      return;
    }
    break;
  default:
    break;
  }

  Diag << "static_cast/const_cast/reinterpret_cast";
}

} // namespace readability
} // namespace google
} // namespace tidy
} // namespace clang
