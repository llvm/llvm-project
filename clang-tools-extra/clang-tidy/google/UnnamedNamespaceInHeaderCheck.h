//===--- UnnamedNamespaceInHeaderCheck.h - clang-tidy -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_UNNAMEDNAMESPACEINHEADERCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_UNNAMEDNAMESPACEINHEADERCHECK_H

#include "../ClangTidyCheck.h"
#include "../utils/FileExtensionsUtils.h"

namespace clang::tidy::google::build {

/// Finds anonymous namespaces in headers.
///
/// https://google.github.io/styleguide/cppguide.html#Namespaces
///
/// Corresponding cpplint.py check name: 'build/namespaces'.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/google/build-namespaces.html
class UnnamedNamespaceInHeaderCheck : public ClangTidyCheck {
public:
  UnnamedNamespaceInHeaderCheck(StringRef Name, ClangTidyContext *Context);
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  FileExtensionsSet HeaderFileExtensions;
};

} // namespace clang::tidy::google::build

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_UNNAMEDNAMESPACEINHEADERCHECK_H
