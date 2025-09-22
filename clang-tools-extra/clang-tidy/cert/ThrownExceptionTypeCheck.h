//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CERT_THROWNEXCEPTIONTYPECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CERT_THROWNEXCEPTIONTYPECHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::cert {

/// Checks whether a thrown object is nothrow copy constructible.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/cert/err60-cpp.html
class ThrownExceptionTypeCheck : public ClangTidyCheck {
public:
  ThrownExceptionTypeCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::cert

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CERT_THROWNEXCEPTIONTYPECHECK_H
