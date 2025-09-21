//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VariadicFunctionDefCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::cert {

void VariadicFunctionDefCheck::registerMatchers(MatchFinder *Finder) {
  // We only care about function *definitions* that are variadic, and do not
  // have extern "C" language linkage.
  Finder->addMatcher(
      functionDecl(isDefinition(), isVariadic(), unless(isExternC()))
          .bind("func"),
      this);
}

void VariadicFunctionDefCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *FD = Result.Nodes.getNodeAs<FunctionDecl>("func");

  diag(FD->getLocation(),
       "do not define a C-style variadic function; consider using a function "
       "parameter pack or currying instead");
}

} // namespace clang::tidy::cert
