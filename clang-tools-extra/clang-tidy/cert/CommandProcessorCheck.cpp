//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CommandProcessorCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::cert {

void CommandProcessorCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      callExpr(
          callee(functionDecl(hasAnyName("::system", "::popen", "::_popen"))
                     .bind("func")),
          // Do not diagnose when the call expression passes a null pointer
          // constant to system(); that only checks for the presence of a
          // command processor, which is not a security risk by itself.
          unless(callExpr(callee(functionDecl(hasName("::system"))),
                          argumentCountIs(1),
                          hasArgument(0, nullPointerConstant()))))
          .bind("expr"),
      this);
}

void CommandProcessorCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Fn = Result.Nodes.getNodeAs<FunctionDecl>("func");
  const auto *E = Result.Nodes.getNodeAs<CallExpr>("expr");

  diag(E->getExprLoc(), "calling %0 uses a command processor") << Fn;
}

} // namespace clang::tidy::cert
