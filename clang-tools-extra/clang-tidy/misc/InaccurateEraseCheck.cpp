//===--- InaccurateEraseCheck.cpp - clang-tidy-----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "InaccurateEraseCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace misc {

void InaccurateEraseCheck::registerMatchers(MatchFinder *Finder) {
  // Only register the matchers for C++; the functionality currently does not
  // provide any benefit to other languages, despite being benign.
  if (!getLangOpts().CPlusPlus)
    return;

  const auto CheckForEndCall = hasArgument(
      1, anyOf(cxxConstructExpr(has(ignoringParenImpCasts(
                   cxxMemberCallExpr(callee(cxxMethodDecl(hasName("end"))))
                       .bind("InaccEndCall")))),
               anything()));

  Finder->addMatcher(
      cxxMemberCallExpr(
          on(hasType(namedDecl(matchesName("^::std::")))),
          callee(cxxMethodDecl(hasName("erase"))), argumentCountIs(1),
          hasArgument(0, has(ignoringParenImpCasts(
                             callExpr(callee(functionDecl(matchesName(
                                          "^::std::(remove(_if)?|unique)$"))),
                                      CheckForEndCall)
                                 .bind("InaccAlgCall")))),
          unless(isInTemplateInstantiation()))
          .bind("InaccErase"),
      this);
}

void InaccurateEraseCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MemberCall =
      Result.Nodes.getNodeAs<CXXMemberCallExpr>("InaccErase");
  const auto *EndExpr =
      Result.Nodes.getNodeAs<CXXMemberCallExpr>("InaccEndCall");
  const SourceLocation Loc = MemberCall->getLocStart();

  FixItHint Hint;

  if (!Loc.isMacroID() && EndExpr) {
    const auto *AlgCall = Result.Nodes.getNodeAs<CallExpr>("InaccAlgCall");
    std::string ReplacementText = Lexer::getSourceText(
        CharSourceRange::getTokenRange(EndExpr->getSourceRange()),
        *Result.SourceManager, getLangOpts());
    const SourceLocation EndLoc = Lexer::getLocForEndOfToken(
        AlgCall->getLocEnd(), 0, *Result.SourceManager, getLangOpts());
    Hint = FixItHint::CreateInsertion(EndLoc, ", " + ReplacementText);
  }

  diag(Loc, "this call will remove at most one item even when multiple items "
            "should be removed")
      << Hint;
}

} // namespace misc
} // namespace tidy
} // namespace clang
