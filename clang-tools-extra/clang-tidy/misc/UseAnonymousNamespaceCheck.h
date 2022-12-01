//===--- UseAnonymousNamespaceCheck.h - clang-tidy --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_USEANONYMOUSNAMESPACECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_USEANONYMOUSNAMESPACECHECK_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace misc {

/// Warns when using 'static' functions or variables at global scope, and
/// suggests moving them to an anonymous namespace. It also suggests removing
/// 'static' if they are already inside an anonymous namespace.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/modernize/use-anonymous-namespace.html
class UseAnonymousNamespaceCheck : public ClangTidyCheck {
public:
  UseAnonymousNamespaceCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  template <typename T> void processMatch(const T *MatchedDecl);
};

} // namespace misc
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_USEANONYMOUSNAMESPACECHECK_H
