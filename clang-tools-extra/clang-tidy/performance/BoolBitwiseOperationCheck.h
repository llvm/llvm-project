//===--- BoolBitwiseOperationCheck.h - clang-tidy ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_BOOLBITWISEOPERATIONCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_BOOLBITWISEOPERATIONCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::performance {

/// Finds potentially unefficient uses of bitwise operators such as `|`,
/// `&` and their compound analogues with `bool` type and suggests replacing
/// them with logical ones, like `||` and `&&`.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/performance/bool-bitwise-operation.html
class BoolBitwiseOperationCheck : public ClangTidyCheck {
public:
  BoolBitwiseOperationCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Options) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }

private:
  bool StrictMode;
};

} // namespace clang::tidy::performance

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_BOOLBITWISEOPERATIONCHECK_H
