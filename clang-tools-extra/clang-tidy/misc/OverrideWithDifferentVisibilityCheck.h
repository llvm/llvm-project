//===--- OverrideWithDifferentVisibilityCheck.h - clang-tidy --*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_OVERRIDEWITHDIFFERENTVISIBILITYCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_OVERRIDEWITHDIFFERENTVISIBILITYCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::misc {

/// Finds virtual function overrides with different visibility than the function
/// in the base class.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/misc/override-with-different-visibility.html
class OverrideWithDifferentVisibilityCheck : public ClangTidyCheck {
public:
  enum class ChangeKind { Any, Widening, Narrowing };

  OverrideWithDifferentVisibilityCheck(StringRef Name,
                                       ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }

private:
  ChangeKind DetectVisibilityChange;
  bool CheckDestructors;
  bool CheckOperators;
  std::vector<llvm::StringRef> IgnoredFunctions;
};

} // namespace clang::tidy::misc

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_OVERRIDEWITHDIFFERENTVISIBILITYCHECK_H
