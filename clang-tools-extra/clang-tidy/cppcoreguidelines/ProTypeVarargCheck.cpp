//===--- ProTypeVarargCheck.cpp - clang-tidy-------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProTypeVarargCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace cppcoreguidelines {

const internal::VariadicDynCastAllOfMatcher<Stmt, VAArgExpr> vAArgExpr;

void ProTypeVarargCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus)
    return;

  Finder->addMatcher(vAArgExpr().bind("va_use"), this);

  Finder->addMatcher(
      callExpr(callee(functionDecl(isVariadic()))).bind("callvararg"), this);
}

static bool hasSingleVariadicArgumentWithValue(const CallExpr *C, uint64_t I) {
  const auto *FDecl = dyn_cast<FunctionDecl>(C->getCalleeDecl());
  if (!FDecl)
    return false;

  auto N = FDecl->getNumParams(); // Number of parameters without '...'
  if (C->getNumArgs() != N + 1)
    return false; // more/less than one argument passed to '...'

  const auto *IntLit =
      dyn_cast<IntegerLiteral>(C->getArg(N)->IgnoreParenImpCasts());
  if (!IntLit)
    return false;

  if (IntLit->getValue() != I)
    return false;

  return true;
}

void ProTypeVarargCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *Matched = Result.Nodes.getNodeAs<CallExpr>("callvararg")) {
    if (hasSingleVariadicArgumentWithValue(Matched, 0))
      return;
    diag(Matched->getExprLoc(), "do not call c-style vararg functions");
  }

  if (const auto *Matched = Result.Nodes.getNodeAs<Expr>("va_use")) {
    diag(Matched->getExprLoc(),
         "do not use va_start/va_arg to define c-style vararg functions; "
         "use variadic templates instead");
  }

  if (const auto *Matched = Result.Nodes.getNodeAs<VarDecl>("va_list")) {
    auto SR = Matched->getSourceRange();
    if (SR.isInvalid())
      return; // some implicitly generated builtins take va_list
    diag(SR.getBegin(), "do not declare variables of type va_list; "
                        "use variadic templates instead");
  }
}

} // namespace cppcoreguidelines
} // namespace tidy
} // namespace clang
