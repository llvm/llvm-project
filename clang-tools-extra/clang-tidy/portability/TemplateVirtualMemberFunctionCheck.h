//===--- TemplateVirtualMemberFunctionCheck.h - clang-tidy ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PORTABILITY_TEMPLATEVIRTUALMEMBERFUNCTIONCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PORTABILITY_TEMPLATEVIRTUALMEMBERFUNCTIONCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::portability {

/// Detects C++ [temp.inst#11]: "It is unspecified whether or not an
/// implementation implicitly instantiates a virtual member function of a class
/// template if the virtual member function would not otherwise be
/// instantiated."
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/portability/template-virtual-member-function.html
class TemplateVirtualMemberFunctionCheck : public ClangTidyCheck {
public:
  TemplateVirtualMemberFunctionCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
};

} // namespace clang::tidy::portability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PORTABILITY_TEMPLATEVIRTUALMEMBERFUNCTIONCHECK_H
