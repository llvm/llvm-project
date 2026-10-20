//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_STDNAMESPACEMODIFICATIONCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_STDNAMESPACEMODIFICATIONCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::bugprone {

/// Modification of the std or posix namespace can result in undefined behavior.
/// This check warns for such modifications.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/bugprone/std-namespace-modification.html
class StdNamespaceModificationCheck : public ClangTidyCheck {
public:
  StdNamespaceModificationCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_STDNAMESPACEMODIFICATIONCHECK_H
