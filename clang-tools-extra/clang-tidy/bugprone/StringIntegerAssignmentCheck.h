//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_STRINGINTEGERASSIGNMENTCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_STRINGINTEGERASSIGNMENTCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::bugprone {

/// Finds instances where an integer is assigned to a string.
///
/// For more details see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone/string-integer-assignment.html
class StringIntegerAssignmentCheck : public ClangTidyCheck {
public:
  StringIntegerAssignmentCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_STRINGINTEGERASSIGNMENTCHECK_H
