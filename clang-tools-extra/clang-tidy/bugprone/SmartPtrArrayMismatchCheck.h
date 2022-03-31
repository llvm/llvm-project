//===--- SharedPtrArrayMismatchCheck.h - clang-tidy -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SMARTPTRARRAYMISMATCHCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SMARTPTRARRAYMISMATCHCHECK_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace bugprone {

/// Find constructions of smart (unique or shared) pointers where the pointer
/// is declared with non-array target type and an array (created with a
/// new-expression) is passed to it.
class SmartPtrArrayMismatchCheck : public ClangTidyCheck {
public:
  SmartPtrArrayMismatchCheck(StringRef Name, ClangTidyContext *Context,
                             StringRef SmartPointerName);

  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus11;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

protected:
  using SmartPtrClassMatcher = ast_matchers::internal::BindableMatcher<Decl>;

  /// Returns matcher that match with different smart pointer classes.
  ///
  /// Requires to bind pointer type (qualType) with PointerTypeN string declared
  /// in this class.
  virtual SmartPtrClassMatcher getSmartPointerClassMatcher() const = 0;

  static const char PointerTypeN[];

private:
  StringRef const SmartPointerName;
};

} // namespace bugprone
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SMARTPTRARRAYMISMATCHCHECK_H
