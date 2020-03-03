//===--- UseDefaultMemberInitCheck.h - clang-tidy----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USE_DEFAULT_MEMBER_INIT_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USE_DEFAULT_MEMBER_INIT_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace modernize {

/// Convert a default constructor's member initializers into default member
/// initializers.  Remove member initializers that are the same as a default
/// member initializer.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/modernize-use-default-member-init.html
class UseDefaultMemberInitCheck : public ClangTidyCheck {
public:
  UseDefaultMemberInitCheck(StringRef Name, ClangTidyContext *Context);
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus11;
  }
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  void checkDefaultInit(const ast_matchers::MatchFinder::MatchResult &Result,
                        const CXXCtorInitializer *Init);
  void checkExistingInit(const ast_matchers::MatchFinder::MatchResult &Result,
                         const CXXCtorInitializer *Init);

  const bool UseAssignment;
  const bool IgnoreMacros;
};

} // namespace modernize
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USE_DEFAULT_MEMBER_INIT_H
