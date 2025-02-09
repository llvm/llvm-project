//===--- NlohmannJsonExplicitConversionsCheck.h - clang-tidy ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_NLOHMANNJSONEXPLICITCONVERSIONSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_NLOHMANNJSONEXPLICITCONVERSIONSCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::modernize {

/// Convert implicit conversions via operator ValueType() from nlohmann::json to
/// explicit calls to the get<> method because the next major version of the
/// library will remove support for implicit conversions.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/modernize/nlohmann-json-explicit-conversions.html
class NlohmannJsonExplicitConversionsCheck : public ClangTidyCheck {
public:
  NlohmannJsonExplicitConversionsCheck(StringRef Name,
                                       ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
};

} // namespace clang::tidy::modernize

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_NLOHMANNJSONEXPLICITCONVERSIONSCHECK_H
