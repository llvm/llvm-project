//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_FLOATTYPESCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_FLOATTYPESCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::google::runtime {

/// Finds usages of `long double` and suggests against their use due to lack
/// of portability.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/google/runtime-float.html
class RuntimeFloatCheck : public ClangTidyCheck {
public:
  RuntimeFloatCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus && !LangOpts.ObjC;
  }
};

} // namespace clang::tidy::google::runtime

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_FLOATTYPESCHECK_H
