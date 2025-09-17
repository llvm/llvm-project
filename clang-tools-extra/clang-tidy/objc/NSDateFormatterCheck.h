//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_OBJC_NSDATEFORMATTERCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_OBJC_NSDATEFORMATTERCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::objc {

/// Checks the string pattern used as a date format specifier and reports
/// warnings if it contains any incorrect sub-pattern.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/objc/nsdate-formatter.html
class NSDateFormatterCheck : public ClangTidyCheck {
public:
  NSDateFormatterCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.ObjC;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::objc

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_OBJC_NSDATEFORMATTERCHECK_H
