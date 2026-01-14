//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "static_in_constexpr.hpp"
#include "clang/AST/Decl.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace libcpp {

void static_in_constexpr::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(varDecl(isStaticLocal()).bind("var"), this);
}

void static_in_constexpr::check(const MatchFinder::MatchResult &Result) {
  const auto *Var = Result.Nodes.getNodeAs<clang::VarDecl>("var");
  if (!Var)
      return;

  const clang::DeclContext *DC = Var->getDeclContext();

  while (DC && !DC->isFunctionOrMethod())
    DC = DC->getParent();

  const auto *FD = llvm::dyn_cast_or_null<clang::FunctionDecl>(DC);
  if (!FD)
    return;

  if (FD->isConstexpr()) {
    diag(Var->getLocation(),
         "variable of static or thread storage duration inside constexpr "
         "function");
  }
}

} // namespace libcpp
