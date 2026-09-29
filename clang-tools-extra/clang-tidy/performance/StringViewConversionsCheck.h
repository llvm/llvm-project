//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_STRINGVIEWCONVERSIONSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_STRINGVIEWCONVERSIONSCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::performance {

/// Finds and removes redundant conversions from std::string_view to std::string
/// in call expressions expecting std::string_view.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/performance/string-view-conversions.html
class StringViewConversionsCheck : public ClangTidyCheck {
public:
  StringViewConversionsCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus17;
  }
};

} // namespace clang::tidy::performance

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_STRINGVIEWCONVERSIONSCHECK_H
