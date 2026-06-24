//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UnsafeApiFunctionsCallsCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

void UnsafeApiFunctionsCallsCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      callExpr(callee(functionDecl(hasAnyName("setvbuf", "::std::setvbuf",
                                              "setbuf", "::std::setbuf"))))
          .bind("call"),
      this);
}

void UnsafeApiFunctionsCallsCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Call = Result.Nodes.getNodeAs<CallExpr>("call");
  if (!Call || Call->getNumArgs() < 2)
    return;

  const Expr *BufArg = Call->getArg(1)->IgnoreParenImpCasts();

  // NULL is fine (used for _IONBF with setvbuf, or to disable buffering with
  // setbuf).
  if (BufArg->isNullPointerConstant(*Result.Context,
                                    Expr::NPC_ValueDependentIsNotNull))
    return;

  // Resolve to the underlying VarDecl if it's a DeclRefExpr.
  const VarDecl *VD = nullptr;
  if (const auto *DRE = dyn_cast<DeclRefExpr>(BufArg))
    VD = dyn_cast<VarDecl>(DRE->getDecl());
  else if (const auto *UO = dyn_cast<UnaryOperator>(BufArg)) {
    // Handle &buf[0]
    if (UO->getOpcode() == UO_AddrOf) {
      if (const auto *ASE = dyn_cast<ArraySubscriptExpr>(
              UO->getSubExpr()->IgnoreParenImpCasts())) {
        if (const auto *DRE =
                dyn_cast<DeclRefExpr>(ASE->getBase()->IgnoreParenImpCasts()))
          VD = dyn_cast<VarDecl>(DRE->getDecl());
      }
    }
  }

  if (!VD)
    return;

  // Only warn for local automatic (stack) variables.
  if (!VD->isLocalVarDecl() || VD->isStaticLocal())
    return;

  // Check if the variable is an array type (direct stack buffer).
  // For pointer variables, they could point to malloc'd memory — don't warn.
  if (!VD->getType()->isArrayType())
    return;

  // Get the function name for the diagnostic message.
  const auto *Callee = Call->getDirectCallee();
  StringRef FuncName = Callee ? Callee->getName() : "setvbuf";

  diag(Call->getBeginLoc(),
       "passing stack-allocated buffer to '%0'; buffer must outlive the "
       "stream; use a static, global, or dynamically allocated buffer instead")
      << FuncName << Call->getArg(1)->getSourceRange();
}

} // namespace clang::tidy::bugprone
