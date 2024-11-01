//===--- InaccurateEraseCheck.cpp - clang-tidy-----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InaccurateEraseCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

void InaccurateEraseCheck::registerMatchers(MatchFinder *Finder) {
  const auto EndCall =
      callExpr(
          callee(functionDecl(hasAnyName("remove", "remove_if", "unique"))),
          hasArgument(
              1, optionally(cxxMemberCallExpr(callee(cxxMethodDecl(hasName("end"))))
                           .bind("end"))))
          .bind("alg");

  const auto DeclInStd = type(hasUnqualifiedDesugaredType(
      tagType(hasDeclaration(decl(isInStdNamespace())))));
  Finder->addMatcher(
      cxxMemberCallExpr(
          on(anyOf(hasType(DeclInStd), hasType(pointsTo(DeclInStd)))),
          callee(cxxMethodDecl(hasName("erase"))), argumentCountIs(1),
          hasArgument(0, EndCall))
          .bind("erase"),
      this);
}

void InaccurateEraseCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MemberCall =
      Result.Nodes.getNodeAs<CXXMemberCallExpr>("erase");
  const auto *EndExpr =
      Result.Nodes.getNodeAs<CXXMemberCallExpr>("end");
  const SourceLocation Loc = MemberCall->getBeginLoc();

  FixItHint Hint;

  if (!Loc.isMacroID() && EndExpr) {
    const auto *AlgCall = Result.Nodes.getNodeAs<CallExpr>("alg");
    std::string ReplacementText = std::string(Lexer::getSourceText(
        CharSourceRange::getTokenRange(EndExpr->getSourceRange()),
        *Result.SourceManager, getLangOpts()));
    const SourceLocation EndLoc = Lexer::getLocForEndOfToken(
        AlgCall->getEndLoc(), 0, *Result.SourceManager, getLangOpts());
    Hint = FixItHint::CreateInsertion(EndLoc, ", " + ReplacementText);
  }

  diag(Loc, "this call will remove at most one item even when multiple items "
            "should be removed")
      << Hint;
}

} // namespace clang::tidy::bugprone
