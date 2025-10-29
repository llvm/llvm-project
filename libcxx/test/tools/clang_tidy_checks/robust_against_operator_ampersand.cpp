//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-tidy/ClangTidyCheck.h"
#include "clang-tidy/ClangTidyModuleRegistry.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/FixIt.h"

#include "robust_against_operator_ampersand.hpp"

// This clang-tidy check ensures that we don't use operator& on dependant
// types. If the type is user supplied it may call the type's operator&.
// Instead use std::addressof.
//
// This is part of libc++'s policy
// https://libcxx.llvm.org/CodingGuidelines.html#don-t-use-argument-dependent-lookup-unless-required-by-the-standard

// TODO(LLVM-21) Remove dependentScopeDeclRefExpr
// dependentScopeDeclRefExpr requires Clang 20, this uses the same definition as Clang
#if defined(__clang_major__) && __clang_major__ < 20
namespace clang::ast_matchers {
const internal::VariadicDynCastAllOfMatcher<Stmt, DependentScopeDeclRefExpr> dependentScopeDeclRefExpr;
} // namespace clang::ast_matchers
#endif

namespace libcpp {
robust_against_operator_ampersand::robust_against_operator_ampersand(
    llvm::StringRef name, clang::tidy::ClangTidyContext* context)
    : clang::tidy::ClangTidyCheck(name, context) {}

void robust_against_operator_ampersand::registerMatchers(clang::ast_matchers::MatchFinder* finder) {
  using namespace clang::ast_matchers;
  finder->addMatcher(
      cxxOperatorCallExpr(allOf(hasOperatorName("&"), argumentCountIs(1), isTypeDependent()),
                          unless(hasUnaryOperand(dependentScopeDeclRefExpr())))
          .bind("match"),
      this);
}

void robust_against_operator_ampersand::check(const clang::ast_matchers::MatchFinder::MatchResult& result) {
  if (const auto* call = result.Nodes.getNodeAs< clang::CXXOperatorCallExpr >("match"); call != nullptr) {
    diag(call->getBeginLoc(), "Guard against user provided operator& for dependent types.")
        << clang::FixItHint::CreateReplacement(
               call->getSourceRange(),
               (llvm::Twine(
                    "std::addressof(" + clang::tooling::fixit::getText(*call->getArg(0), *result.Context) + ")"))
                   .str());
  }
}

} // namespace libcpp
