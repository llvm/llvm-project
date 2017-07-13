//===--- CloexecSocketCheck.cpp - clang-tidy-------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CloexecSocketCheck.h"
#include "../utils/ASTUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace android {

static constexpr const char *SOCK_CLOEXEC = "SOCK_CLOEXEC";

void CloexecSocketCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      callExpr(callee(functionDecl(isExternC(), returns(isInteger()),
                                   hasName("socket"),
                                   hasParameter(0, hasType(isInteger())),
                                   hasParameter(1, hasType(isInteger())),
                                   hasParameter(2, hasType(isInteger())))
                          .bind("funcDecl")))
          .bind("socketFn"),
      this);
}

void CloexecSocketCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedCall = Result.Nodes.getNodeAs<CallExpr>("socketFn");
  const auto *FD = Result.Nodes.getNodeAs<FunctionDecl>("funcDecl");
  const Expr *FlagArg = MatchedCall->getArg(1);
  SourceManager &SM = *Result.SourceManager;

  if (utils::exprHasBitFlagWithSpelling(FlagArg->IgnoreParenCasts(), SM,
                     Result.Context->getLangOpts(), SOCK_CLOEXEC))
    return;

  SourceLocation EndLoc =
      Lexer::getLocForEndOfToken(SM.getFileLoc(FlagArg->getLocEnd()), 0, SM,
                                 Result.Context->getLangOpts());

  diag(EndLoc, "%0 should use %1 where possible")
      << FD << SOCK_CLOEXEC
      << FixItHint::CreateInsertion(EndLoc,
                                    (Twine(" | ") + SOCK_CLOEXEC).str());
}

} // namespace android
} // namespace tidy
} // namespace clang
