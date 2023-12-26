//===--- UseDigitSeparatorCheck.h - clang-tidy ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USEDIGITSEPARATORCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USEDIGITSEPARATORCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::modernize {

/// The check that looks for long integral constants and inserts the digits
/// separator (') appropriately. Groupings:
///     - decimal integral constants, groups of 3 digits, e.g. int x = 1'000;
///     - binary integral constants, groups of 4 digits, e.g. int x =
///     0b0010'0011;
///     - octal integral constants, groups of 3 digits, e.g. int x = 0377'777;
///     - hexadecimal integral constants, groups of 4 digits, e.g. unsigned long
///     x = 0xffff'0000;
///     - floating-point constants, group into 3 digits on either side of the
///     decimal point, e.g. float x = 3'456.001'25f;
///
/// Requires c++ version 14 or later
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/modernize/use-digit-separator.html
class UseDigitSeparatorCheck : public ClangTidyCheck {
public:
  UseDigitSeparatorCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::modernize

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USEDIGITSEPARATORCHECK_H
