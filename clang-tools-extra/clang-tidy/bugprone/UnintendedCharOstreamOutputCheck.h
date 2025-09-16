//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNINTENDEDCHAROSTREAMOUTPUTCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNINTENDEDCHAROSTREAMOUTPUTCHECK_H

#include "../ClangTidyCheck.h"
#include <optional>

namespace clang::tidy::bugprone {

/// Finds unintended character output from `unsigned char` and `signed char` to
/// an ostream.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone/unintended-char-ostream-output.html
class UnintendedCharOstreamOutputCheck : public ClangTidyCheck {
public:
  UnintendedCharOstreamOutputCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }

private:
  const std::vector<StringRef> AllowedTypes;
  const std::optional<StringRef> CastTypeName;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNINTENDEDCHAROSTREAMOUTPUTCHECK_H
