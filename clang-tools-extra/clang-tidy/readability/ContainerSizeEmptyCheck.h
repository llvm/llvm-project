//===--- ContainerSizeEmptyCheck.h - clang-tidy -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_CONTAINERSIZEEMPTYCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_CONTAINERSIZEEMPTYCHECK_H

#include "../ClangTidyCheck.h"
#include <vector>

namespace clang::tidy::readability {

/// Checks whether a call to the `size()`/`length()` method can be replaced with
/// a call to `empty()`.
///
/// The emptiness of a container should be checked using the `empty()` method
/// instead of the `size()`/`length()` method. It shows clearer intent to use
/// `empty()`. Furthermore some containers (for example, a `std::forward_list`)
/// may implement the `empty()` method but not implement the `size()` or
/// `length()` method. Using `empty()` whenever possible makes it easier to
/// switch to another container in the future.
class ContainerSizeEmptyCheck : public ClangTidyCheck {
public:
  ContainerSizeEmptyCheck(StringRef Name, ClangTidyContext *Context);
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }

private:
  std::vector<llvm::StringRef> ExcludedComparisonTypes;
};

} // namespace clang::tidy::readability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_CONTAINERSIZEEMPTYCHECK_H
