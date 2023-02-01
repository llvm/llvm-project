//===--- RawStringLiteralCheck.h - clang-tidy--------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_RAW_STRING_LITERAL_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_RAW_STRING_LITERAL_H

#include "../ClangTidyCheck.h"
#include <bitset>

namespace clang::tidy::modernize {

using CharsBitSet = std::bitset<1 << CHAR_BIT>;

/// This check replaces string literals with escaped characters to
/// raw string literals.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/modernize/raw-string-literal.html
class RawStringLiteralCheck : public ClangTidyCheck {
public:
  RawStringLiteralCheck(StringRef Name, ClangTidyContext *Context);

  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus11;
  }
  void storeOptions(ClangTidyOptions::OptionMap &Options) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  void replaceWithRawStringLiteral(
      const ast_matchers::MatchFinder::MatchResult &Result,
      const StringLiteral *Literal, StringRef Replacement);

  std::string DelimiterStem;
  CharsBitSet DisallowedChars;
  const bool ReplaceShorterLiterals;
};

} // namespace clang::tidy::modernize

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_RAW_STRING_LITERAL_H
