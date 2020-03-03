//===--- UseEqualsDefaultCheck.h - clang-tidy--------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USE_EQUALS_DEFAULT_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USE_EQUALS_DEFAULT_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace modernize {

/// Replace default bodies of special member functions with '= default;'.
/// \code
///   struct A {
///     A() {}
///     ~A();
///   };
///   A::~A() {}
/// \endcode
/// Is converted to:
/// \code
///   struct A {
///     A() = default;
///     ~A();
///   };
///   A::~A() = default;
/// \endcode
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/modernize-use-equals-default.html
class UseEqualsDefaultCheck : public ClangTidyCheck {
public:
  UseEqualsDefaultCheck(StringRef Name, ClangTidyContext *Context);
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  const bool IgnoreMacros;
};

} // namespace modernize
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USE_EQUALS_DEFAULT_H
