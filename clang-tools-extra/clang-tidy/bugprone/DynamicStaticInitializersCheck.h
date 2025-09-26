//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_DYNAMIC_STATIC_INITIALIZERS_CHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_DYNAMIC_STATIC_INITIALIZERS_CHECK_H

#include "../ClangTidyCheck.h"
#include "../FileExtensionsSet.h"

namespace clang::tidy::bugprone {

/// Finds dynamically initialized static variables in header files.
class DynamicStaticInitializersCheck : public ClangTidyCheck {
public:
  DynamicStaticInitializersCheck(StringRef Name, ClangTidyContext *Context);
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus && !LangOpts.ThreadsafeStatics;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  FileExtensionsSet HeaderFileExtensions;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_DYNAMIC_STATIC_INITIALIZERS_CHECK_H
