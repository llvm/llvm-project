//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_DERIVEDMETHODSHADOWINGBASEMETHODCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_DERIVEDMETHODSHADOWINGBASEMETHODCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::bugprone {

/// Checks that a derived class does not define the same (non virtual) method as
/// a base class
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/bugprone/derived-method-shadowing-base-method.html
class DerivedMethodShadowingBaseMethodCheck : public ClangTidyCheck {
public:
  DerivedMethodShadowingBaseMethodCheck(StringRef Name,
                                        ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_DERIVEDMETHODSHADOWINGBASEMETHODCHECK_H
