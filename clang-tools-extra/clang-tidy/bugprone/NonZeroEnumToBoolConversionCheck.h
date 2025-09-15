//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_NONZEROENUMTOBOOLCONVERSIONCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_NONZEROENUMTOBOOLCONVERSIONCHECK_H

#include "../ClangTidyCheck.h"
#include <vector>

namespace clang::tidy::bugprone {

/// Detect implicit and explicit casts of `enum` type into `bool` where
/// `enum` type doesn't have a zero-value enumerator.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone/non-zero-enum-to-bool-conversion.html
class NonZeroEnumToBoolConversionCheck : public ClangTidyCheck {
public:
  NonZeroEnumToBoolConversionCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override;

private:
  const std::vector<StringRef> EnumIgnoreList;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_NONZEROENUMTOBOOLCONVERSIONCHECK_H
