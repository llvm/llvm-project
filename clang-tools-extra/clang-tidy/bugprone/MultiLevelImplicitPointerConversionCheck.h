//===--- MultiLevelImplicitPointerConversionCheck.h - clang-tidy *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_MULTILEVELIMPLICITPOINTERCONVERSIONCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_MULTILEVELIMPLICITPOINTERCONVERSIONCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::bugprone {

/// Detects implicit conversions between pointers of different levels of
/// indirection.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone/multi-level-implicit-pointer-conversion.html
class MultiLevelImplicitPointerConversionCheck : public ClangTidyCheck {
public:
  MultiLevelImplicitPointerConversionCheck(StringRef Name,
                                           ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  std::optional<TraversalKind> getCheckTraversalKind() const override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return EnableInC ? true : LangOpts.CPlusPlus;
  }

private:
  const bool EnableInC;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_MULTILEVELIMPLICITPOINTERCONVERSIONCHECK_H
