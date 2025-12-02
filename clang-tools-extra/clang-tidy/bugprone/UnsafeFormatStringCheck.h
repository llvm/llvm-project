//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNSAFEFORMATSTRINGCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNSAFEFORMATSTRINGCHECK_H

#include "../ClangTidyCheck.h"
#include "../utils/Matchers.h"

namespace clang::tidy::bugprone {

/// Detects usage of vulnerable format string functions with unbounded %s
/// specifiers that can cause buffer overflows.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/bugprone/unsafe-format-string.html
class UnsafeFormatStringCheck : public ClangTidyCheck {
public:
  UnsafeFormatStringCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

  struct CheckedFunction {
    std::string Name;
    matchers::MatchesAnyListedNameMatcher::NameMatcher Pattern;
    unsigned long FormatStringLocation;
  };

private:
  const std::vector<CheckedFunction> CustomPrintfFunctions;
  const std::vector<CheckedFunction> CustomScanfFunctions;
  static const StringLiteral *
  getFormatLiteral(const CallExpr *,
                   const std::vector<CheckedFunction> &CustomFunctions);
  static bool hasUnboundedStringSpecifier(StringRef Fmt, bool IsScanfFamily);
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNSAFEFORMATSTRINGCHECK_H
