//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USEAGGREGATECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USEAGGREGATECHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::modernize {

/// Finds classes that could be aggregates if their trivial constructors
/// were removed.
///
/// A constructor is considered trivial when it simply forwards each parameter
/// to a member in declaration order and has an empty body. Removing such
/// constructors enables aggregate initialization, which is often clearer and
/// supports designated initializers (C++20).
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/modernize/use-aggregate.html
class UseAggregateCheck : public ClangTidyCheck {
public:
  UseAggregateCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
};

} // namespace clang::tidy::modernize

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USEAGGREGATECHECK_H
