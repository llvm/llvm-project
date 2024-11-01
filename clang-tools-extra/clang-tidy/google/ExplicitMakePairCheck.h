//===--- ExplicitMakePairCheck.h - clang-tidy -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_EXPLICITMAKEPAIRCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_EXPLICITMAKEPAIRCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::google::build {

/// Check that `make_pair`'s template arguments are deduced.
///
/// G++ 4.6 in C++11 mode fails badly if `make_pair`'s template arguments are
/// specified explicitly, and such use isn't intended in any case.
///
/// Corresponding cpplint.py check name: 'build/explicit_make_pair'.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/google/build-explicit-make-pair.html
class ExplicitMakePairCheck : public ClangTidyCheck {
public:
  ExplicitMakePairCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::google::build

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_EXPLICITMAKEPAIRCHECK_H
