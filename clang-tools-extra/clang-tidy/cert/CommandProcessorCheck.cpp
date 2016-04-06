//===--- Env33CCheck.cpp - clang-tidy--------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandProcessorCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace cert {

void CommandProcessorCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      callExpr(
          callee(functionDecl(anyOf(hasName("::system"), hasName("::popen"),
                                    hasName("::_popen")))
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

} // namespace cert
} // namespace tidy
} // namespace clang
