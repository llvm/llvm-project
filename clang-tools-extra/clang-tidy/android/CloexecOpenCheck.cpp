//===--- CloexecOpenCheck.cpp - clang-tidy---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CloexecOpenCheck.h"
#include "../utils/ASTUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace android {

static constexpr const char *O_CLOEXEC = "O_CLOEXEC";

void CloexecOpenCheck::registerMatchers(MatchFinder *Finder) {
  auto CharPointerType = hasType(pointerType(pointee(isAnyCharacter())));

  Finder->addMatcher(
      callExpr(callee(functionDecl(isExternC(), returns(isInteger()),
                                   hasAnyName("open", "open64"),
                                   hasParameter(0, CharPointerType),
                                   hasParameter(1, hasType(isInteger())))
                          .bind("funcDecl")))
          .bind("openFn"),
      this);
  Finder->addMatcher(
      callExpr(callee(functionDecl(isExternC(), returns(isInteger()),
                                   hasName("openat"),
                                   hasParameter(0, hasType(isInteger())),
                                   hasParameter(1, CharPointerType),
                                   hasParameter(2, hasType(isInteger())))
                          .bind("funcDecl")))
          .bind("openatFn"),
      this);
}

void CloexecOpenCheck::check(const MatchFinder::MatchResult &Result) {
  const Expr *FlagArg = nullptr;
  if (const auto *OpenFnCall = Result.Nodes.getNodeAs<CallExpr>("openFn"))
    FlagArg = OpenFnCall->getArg(1);
  else if (const auto *OpenFnCall =
               Result.Nodes.getNodeAs<CallExpr>("openatFn"))
    FlagArg = OpenFnCall->getArg(2);
  assert(FlagArg);

  const auto *FD = Result.Nodes.getNodeAs<FunctionDecl>("funcDecl");

  // Check the required flag.
  SourceManager &SM = *Result.SourceManager;
  if (utils::exprHasBitFlagWithSpelling(FlagArg->IgnoreParenCasts(), SM,
                     Result.Context->getLangOpts(), O_CLOEXEC))
    return;

  SourceLocation EndLoc =
      Lexer::getLocForEndOfToken(SM.getFileLoc(FlagArg->getLocEnd()), 0, SM,
                                 Result.Context->getLangOpts());

  diag(EndLoc, "%0 should use %1 where possible")
      << FD << O_CLOEXEC
      << FixItHint::CreateInsertion(EndLoc, (Twine(" | ") + O_CLOEXEC).str());
}

} // namespace android
} // namespace tidy
} // namespace clang
