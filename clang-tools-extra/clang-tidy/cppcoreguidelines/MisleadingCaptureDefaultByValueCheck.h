//===--- MisleadingCaptureDefaultByValueCheck.h - clang-tidy*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_MISLEADINGCAPTUREDEFAULTBYVALUECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_MISLEADINGCAPTUREDEFAULTBYVALUECHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::cppcoreguidelines {

/// Warns when lambda specify a by-value capture default and capture ``this``.
///
/// By-value capture defaults in member functions can be misleading about
/// whether data members are captured by value or reference.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/cppcoreguidelines/misleading-capture-default-by-value.html
class MisleadingCaptureDefaultByValueCheck : public ClangTidyCheck {
public:
  MisleadingCaptureDefaultByValueCheck(StringRef Name,
                                       ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus11;
  }
};

} // namespace clang::tidy::cppcoreguidelines

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_MISLEADINGCAPTUREDEFAULTBYVALUECHECK_H
