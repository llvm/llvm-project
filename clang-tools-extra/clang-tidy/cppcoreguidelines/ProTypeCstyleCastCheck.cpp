//===--- ProTypeCstyleCastCheck.cpp - clang-tidy---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProTypeCstyleCastCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace cppcoreguidelines {

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

void ProTypeCstyleCastCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      cStyleCastExpr(unless(isInTemplateInstantiation())).bind("cast"), this);
}

void ProTypeCstyleCastCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedCast = Result.Nodes.getNodeAs<CStyleCastExpr>("cast");

  if (MatchedCast->getCastKind() == CK_BitCast ||
      MatchedCast->getCastKind() == CK_LValueBitCast ||
      MatchedCast->getCastKind() == CK_IntegralToPointer ||
      MatchedCast->getCastKind() == CK_PointerToIntegral ||
      MatchedCast->getCastKind() == CK_ReinterpretMemberPointer) {
    diag(MatchedCast->getBeginLoc(),
         "do not use C-style cast to convert between unrelated types");
    return;
  }

  QualType SourceType = MatchedCast->getSubExpr()->getType();

  if (MatchedCast->getCastKind() == CK_BaseToDerived) {
    const auto *SourceDecl = SourceType->getPointeeCXXRecordDecl();
    if (!SourceDecl) // The cast is from object to reference.
      SourceDecl = SourceType->getAsCXXRecordDecl();
    if (!SourceDecl)
      return;

    if (SourceDecl->isPolymorphic()) {
      // Leave type spelling exactly as it was (unlike
      // getTypeAsWritten().getAsString() which would spell enum types 'enum
      // X').
      StringRef DestTypeString = Lexer::getSourceText(
          CharSourceRange::getTokenRange(
              MatchedCast->getLParenLoc().getLocWithOffset(1),
              MatchedCast->getRParenLoc().getLocWithOffset(-1)),
          *Result.SourceManager, getLangOpts());

      auto diag_builder = diag(
          MatchedCast->getBeginLoc(),
          "do not use C-style cast to downcast from a base to a derived class; "
          "use dynamic_cast instead");

      const Expr *SubExpr =
          MatchedCast->getSubExprAsWritten()->IgnoreImpCasts();
      std::string CastText = ("dynamic_cast<" + DestTypeString + ">").str();
      if (!isa<ParenExpr>(SubExpr)) {
        CastText.push_back('(');
        diag_builder << FixItHint::CreateInsertion(
            Lexer::getLocForEndOfToken(SubExpr->getEndLoc(), 0,
                                       *Result.SourceManager, getLangOpts()),
            ")");
      }
      auto ParenRange = CharSourceRange::getTokenRange(
          MatchedCast->getLParenLoc(), MatchedCast->getRParenLoc());
      diag_builder << FixItHint::CreateReplacement(ParenRange, CastText);
    } else {
      diag(
          MatchedCast->getBeginLoc(),
          "do not use C-style cast to downcast from a base to a derived class");
    }
    return;
  }

  if (MatchedCast->getCastKind() == CK_NoOp &&
      needsConstCast(SourceType, MatchedCast->getType())) {
    diag(MatchedCast->getBeginLoc(),
         "do not use C-style cast to cast away constness");
  }
}

} // namespace cppcoreguidelines
} // namespace tidy
} // namespace clang
