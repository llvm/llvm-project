//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_UNDELEGATEDCONSTRUCTOR_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_UNDELEGATEDCONSTRUCTOR_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::bugprone {

/// Finds creation of temporary objects in constructors that look like a
/// function call to another constructor of the same class.
///
/// The user most likely meant to use a delegating constructor or base class
/// initializer.
class UndelegatedConstructorCheck : public ClangTidyCheck {
public:
  UndelegatedConstructorCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus11;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_UNDELEGATEDCONSTRUCTOR_H
