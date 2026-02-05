//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_DEFAULTOPERATORNEWONOVERALIGNEDTYPECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_DEFAULTOPERATORNEWONOVERALIGNEDTYPECHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::bugprone {

/// Checks if an object of type with extended alignment is allocated by using
/// the default operator new.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/bugprone/bugprone-default-operator-new-on-overaligned-type.html
class DefaultOperatorNewOnOveralignedTypeCheck : public ClangTidyCheck {
public:
  DefaultOperatorNewOnOveralignedTypeCheck(StringRef Name,
                                           ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return !LangOpts.CPlusPlus17;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_DEFAULTOPERATORNEWONOVERALIGNEDTYPECHECK_H
