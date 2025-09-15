//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_HICPP_EXCEPTION_BASECLASS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_HICPP_EXCEPTION_BASECLASS_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::hicpp {

/// Check for thrown exceptions and enforce they are all derived from
/// std::exception.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/hicpp/exception-baseclass.html
class ExceptionBaseclassCheck : public ClangTidyCheck {
public:
  ExceptionBaseclassCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::hicpp

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_HICPP_EXCEPTION_BASECLASS_H
