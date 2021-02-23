//===--- MoveConstructorInitCheck.h - clang-tidy-----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_MOVECONSTRUCTORINITCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_MOVECONSTRUCTORINITCHECK_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace performance {

/// The check flags user-defined move constructors that have a ctor-initializer
/// initializing a member or base class through a copy constructor instead of a
/// move constructor.
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/performance-move-constructor-init.html
class MoveConstructorInitCheck : public ClangTidyCheck {
public:
  MoveConstructorInitCheck(StringRef Name, ClangTidyContext *Context);
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus11;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace performance
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_MOVECONSTRUCTORINITCHECK_H
