//===--- UseEqualsDeleteCheck.h - clang-tidy---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USE_EQUALS_DELETE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USE_EQUALS_DELETE_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::modernize {

/// Mark unimplemented private special member functions with '= delete'.
/// \code
///   struct A {
///   private:
///     A(const A&);
///     A& operator=(const A&);
///   };
/// \endcode
/// Is converted to:
/// \code
///   struct A {
///   private:
///     A(const A&) = delete;
///     A& operator=(const A&) = delete;
///   };
/// \endcode
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/modernize/use-equals-delete.html
class UseEqualsDeleteCheck : public ClangTidyCheck {
public:
  UseEqualsDeleteCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context),
        IgnoreMacros(Options.getLocalOrGlobal("IgnoreMacros", true)) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  const bool IgnoreMacros;
};

} // namespace clang::tidy::modernize

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USE_EQUALS_DELETE_H
