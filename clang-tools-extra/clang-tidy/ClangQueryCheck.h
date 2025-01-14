//===--- ClangQueryCheck.h - clang-tidy --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANGQUERYCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANGQUERYCHECK_H

#include "ClangTidyCheck.h"
#include "clang/ASTMatchers/Dynamic/VariantValue.h"
#include <vector>

namespace clang::query {
class QuerySession;
} // namespace clang::query

namespace clang::tidy::misc {

/// A check that matches a given matchers printing their binds as warnings
class ClangQueryCheck : public ClangTidyCheck {
  using MatcherVec = std::vector<ast_matchers::dynamic::DynTypedMatcher>;

public:
  ClangQueryCheck(StringRef Name, ClangTidyContext *Context,
                  MatcherVec Matchers)
      : ClangTidyCheck(Name, Context), Matchers(std::move(Matchers)) {}

  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }

private:
  MatcherVec Matchers;
};

} // namespace clang::tidy::misc

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANGQUERYCHECK_H
