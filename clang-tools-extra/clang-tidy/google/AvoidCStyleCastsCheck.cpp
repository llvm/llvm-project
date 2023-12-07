//===--- AvoidCStyleCastsCheck.cpp - clang-tidy -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AvoidCStyleCastsCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::google::readability {

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
  Finder->addMatcher(
      cxxFunctionalCastExpr(unless(hasDescendant(cxxConstructExpr())),
                            unless(hasDescendant(initListExpr())))
          .bind("cast"),
      this);
}

static bool needsConstCast(QualType SourceType, QualType DestType) {
  while ((SourceType->isPointerType() && DestType->isPointerType()) ||
         (SourceType->isReferenceType() && DestType->isReferenceType())) {
    SourceType = SourceType->getPointeeType();
    DestType = DestType->getPointeeType();
    if (SourceType.isConstQualified() && !DestType.isConstQualified()) {
      return (SourceType->isPointerType() == DestType->isPointerType()) &&
             (SourceType->isReferenceType() == DestType->isReferenceType());
    }
  }
  return false;
}

static bool pointedUnqualifiedTypesAreEqual(QualType T1, QualType T2) {
  while ((T1->isPointerType() && T2->isPointerType()) ||
         (T1->isReferenceType() && T2->isReferenceType())) {
    T1 = T1->getPointeeType();
    T2 = T2->getPointeeType();
  }
  return T1.getUnqualifiedType() == T2.getUnqualifiedType();
}

static clang::CharSourceRange getReplaceRange(const ExplicitCastExpr *Expr) {
  if (const auto *CastExpr = dyn_cast<CStyleCastExpr>(Expr))
    return CharSourceRange::getCharRange(
        CastExpr->getLParenLoc(),
        CastExpr->getSubExprAsWritten()->getBeginLoc());
  if (const auto *CastExpr = dyn_cast<CXXFunctionalCastExpr>(Expr))
    return CharSourceRange::getCharRange(CastExpr->getBeginLoc(),
                                         CastExpr->getLParenLoc());
  llvm_unreachable("Unsupported CastExpr");
}

static StringRef getDestTypeString(const SourceManager &SM,
                                   const LangOptions &LangOpts,
                                   const ExplicitCastExpr *Expr) {
  SourceLocation BeginLoc;
  SourceLocation EndLoc;

  if (const auto *CastExpr = dyn_cast<CStyleCastExpr>(Expr)) {
    BeginLoc = CastExpr->getLParenLoc().getLocWithOffset(1);
    EndLoc = CastExpr->getRParenLoc().getLocWithOffset(-1);
  } else if (const auto *CastExpr = dyn_cast<CXXFunctionalCastExpr>(Expr)) {
    BeginLoc = CastExpr->getBeginLoc();
    EndLoc = CastExpr->getLParenLoc().getLocWithOffset(-1);
  } else
    llvm_unreachable("Unsupported CastExpr");

  return Lexer::getSourceText(CharSourceRange::getTokenRange(BeginLoc, EndLoc),
                              SM, LangOpts);
}

void AvoidCStyleCastsCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *CastExpr = Result.Nodes.getNodeAs<ExplicitCastExpr>("cast");

  // Ignore casts in macros.
  if (CastExpr->getExprLoc().isMacroID())
    return;

  // Casting to void is an idiomatic way to mute "unused variable" and similar
  // warnings.
  if (CastExpr->getCastKind() == CK_ToVoid)
    return;

  auto IsFunction = [](QualType T) {
    T = T.getCanonicalType().getNonReferenceType();
    return T->isFunctionType() || T->isFunctionPointerType() ||
           T->isMemberFunctionPointerType();
  };

  const QualType DestTypeAsWritten =
      CastExpr->getTypeAsWritten().getUnqualifiedType();
  const QualType SourceTypeAsWritten =
      CastExpr->getSubExprAsWritten()->getType().getUnqualifiedType();
  const QualType SourceType = SourceTypeAsWritten.getCanonicalType();
  const QualType DestType = DestTypeAsWritten.getCanonicalType();

  CharSourceRange ReplaceRange = getReplaceRange(CastExpr);

  bool FnToFnCast =
      IsFunction(SourceTypeAsWritten) && IsFunction(DestTypeAsWritten);

  const bool ConstructorCast = !CastExpr->getTypeAsWritten().hasQualifiers() &&
      DestTypeAsWritten->isRecordType() &&
      !DestTypeAsWritten->isElaboratedTypeSpecifier();

  if (CastExpr->getCastKind() == CK_NoOp && !FnToFnCast) {
    // Function pointer/reference casts may be needed to resolve ambiguities in
    // case of overloaded functions, so detection of redundant casts is trickier
    // in this case. Don't emit "redundant cast" warnings for function
    // pointer/reference types.
    QualType Src = SourceTypeAsWritten, Dst = DestTypeAsWritten;
    if (const auto *ElTy = dyn_cast<ElaboratedType>(Src))
      Src = ElTy->getNamedType();
    if (const auto *ElTy = dyn_cast<ElaboratedType>(Dst))
      Dst = ElTy->getNamedType();
    if (Src == Dst) {
      diag(CastExpr->getBeginLoc(), "redundant cast to the same type")
          << FixItHint::CreateRemoval(ReplaceRange);
      return;
    }
  }

  // The rest of this check is only relevant to C++.
  // We also disable it for Objective-C++.
  if (!getLangOpts().CPlusPlus || getLangOpts().ObjC)
    return;
  // Ignore code inside extern "C" {} blocks.
  if (!match(expr(hasAncestor(linkageSpecDecl())), *CastExpr, *Result.Context)
           .empty())
    return;
  // Ignore code in .c files and headers included from them, even if they are
  // compiled as C++.
  if (getCurrentMainFile().endswith(".c"))
    return;

  SourceManager &SM = *Result.SourceManager;

  // Ignore code in .c files #included in other files (which shouldn't be done,
  // but people still do this for test and other purposes).
  if (SM.getFilename(SM.getSpellingLoc(CastExpr->getBeginLoc())).endswith(".c"))
    return;

  // Leave type spelling exactly as it was (unlike
  // getTypeAsWritten().getAsString() which would spell enum types 'enum X').
  StringRef DestTypeString = getDestTypeString(SM, getLangOpts(), CastExpr);

  auto Diag =
      diag(CastExpr->getBeginLoc(), "C-style casts are discouraged; use %0");

  auto ReplaceWithCast = [&](std::string CastText) {
    const Expr *SubExpr = CastExpr->getSubExprAsWritten()->IgnoreImpCasts();
    if (!isa<ParenExpr>(SubExpr) && !isa<CXXFunctionalCastExpr>(CastExpr)) {
      CastText.push_back('(');
      Diag << FixItHint::CreateInsertion(
          Lexer::getLocForEndOfToken(SubExpr->getEndLoc(), 0, SM,
                                     getLangOpts()),
          ")");
    }
    Diag << FixItHint::CreateReplacement(ReplaceRange, CastText);
  };
  auto ReplaceWithNamedCast = [&](StringRef CastType) {
    Diag << CastType;
    ReplaceWithCast((CastType + "<" + DestTypeString + ">").str());
  };
  auto ReplaceWithConstructorCall = [&]() {
    Diag << "constructor call syntax";
    // FIXME: Validate DestTypeString, maybe.
    ReplaceWithCast(DestTypeString.str());
  };
  // Suggest appropriate C++ cast. See [expr.cast] for cast notation semantics.
  switch (CastExpr->getCastKind()) {
  case CK_FunctionToPointerDecay:
    ReplaceWithNamedCast("static_cast");
    return;
  case CK_ConstructorConversion:
    if (ConstructorCast) {
      ReplaceWithConstructorCall();
    } else {
      ReplaceWithNamedCast("static_cast");
    }
    return;
  case CK_NoOp:
    if (FnToFnCast) {
      ReplaceWithNamedCast("static_cast");
      return;
    }
    if (SourceType == DestType) {
      Diag << "static_cast (if needed, the cast may be redundant)";
      ReplaceWithCast(("static_cast<" + DestTypeString + ">").str());
      return;
    }
    if (needsConstCast(SourceType, DestType) &&
        pointedUnqualifiedTypesAreEqual(SourceType, DestType)) {
      ReplaceWithNamedCast("const_cast");
      return;
    }
    if (ConstructorCast) {
      ReplaceWithConstructorCall();
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
    [[fallthrough]];
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
      if (SourceType->isVoidPointerType())
        ReplaceWithNamedCast("static_cast");
      else
        ReplaceWithNamedCast("reinterpret_cast");
      return;
    }
    break;
  default:
    break;
  }

  Diag << "static_cast/const_cast/reinterpret_cast";
}

} // namespace clang::tidy::google::readability
