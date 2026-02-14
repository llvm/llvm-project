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

void static_in_constexpr::registerMatchers(MatchFinder* finder) {
  finder->addMatcher(functionDecl(hasDescendant(varDecl(isStaticLocal()).bind("var")), isConstexpr()), this);
}

void static_in_constexpr::check(const MatchFinder::MatchResult& result) {
  if (const auto* var_decl = result.Nodes.getNodeAs<clang::VarDecl>("var"))
    diag(var_decl->getLocation(), "static variables inside constexpr functions aren't supported by GCC before C++23");
}

} // namespace libcpp
