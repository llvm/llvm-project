//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_PREFERSTATICOVERANONYMOUSNAMESPACECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_PREFERSTATICOVERANONYMOUSNAMESPACECHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::llvm_check {

/// Finds function and variable declarations inside anonymous namespace and
/// suggests replacing them with ``static`` declarations.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/llvm/prefer-static-over-anonymous-namespace.html
class PreferStaticOverAnonymousNamespaceCheck : public ClangTidyCheck {
public:
  PreferStaticOverAnonymousNamespaceCheck(StringRef Name,
                                          ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }

private:
  const bool AllowVariableDeclarations;
  const bool AllowMemberFunctionsInClass;
};

} // namespace clang::tidy::llvm_check

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_PREFERSTATICOVERANONYMOUSNAMESPACECHECK_H
