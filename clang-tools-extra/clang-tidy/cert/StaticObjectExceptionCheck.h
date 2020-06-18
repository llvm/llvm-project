//===--- StaticObjectExceptionCheck.h - clang-tidy---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CERT_ERR58_CPP_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CERT_ERR58_CPP_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace cert {

/// Checks whether the constructor for a static or thread_local object will
/// throw.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/cert-err58-cpp.html
class StaticObjectExceptionCheck : public ClangTidyCheck {
public:
  StaticObjectExceptionCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return getLangOpts().CPlusPlus && getLangOpts().CXXExceptions;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace cert
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CERT_ERR58_CPP_H
