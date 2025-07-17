//===--- ConstexprNonStaticInScopeCheck.cpp - clang-tidy ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ConstexprNonStaticInScopeCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::performance {

void ConstexprNonStaticInScopeCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(varDecl(
          hasLocalStorage(),
          isConstexpr(),
          unless(isStaticLocal())
      ).bind("constexprVar"), this);
}

void ConstexprNonStaticInScopeCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "WarnInConstexprFuncCpp23", WarnInConstexprFuncCpp23);
}

void ConstexprNonStaticInScopeCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Var = Result.Nodes.getNodeAs<clang::VarDecl>("constexprVar");
  if (!Var)
    return;
  const auto *EnclosingFunc = llvm::dyn_cast_or_null<clang::FunctionDecl>(Var->getDeclContext());

  if (EnclosingFunc && EnclosingFunc->isConstexpr()) {
    // If the function is constexpr, only warn in C++23 and above
    if (!Result.Context->getLangOpts().CPlusPlus23 || !WarnInConstexprFuncCpp23)
      return; // Don't warn unless in C++23+ AND the option is enabled
  }

  diag(Var->getLocation(),
       "constexpr variable in function scope should be static to ensure static lifetime")
      << FixItHint::CreateInsertion(Var->getSourceRange().getBegin(), "static ");
}

} // namespace clang::tidy::performance
