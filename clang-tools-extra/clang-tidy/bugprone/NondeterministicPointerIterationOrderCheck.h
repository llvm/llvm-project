//=== NondeterministicPointerIterationOrderCheck.h - clang-tidy -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_NONDETERMINISTIC_POINTER_ITERATION_ORDER_CHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_NONDETERMINISTIC_POINTER_ITERATION_ORDER_CHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::bugprone {

/// Finds nondeterministic usages of pointers in unordered containers. The
/// check also finds calls to sorting-like algorithms on a container of
/// pointers.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone/nondeterministic-pointer-iteration-order.html
class NondeterministicPointerIterationOrderCheck : public ClangTidyCheck {
public:
  NondeterministicPointerIterationOrderCheck(StringRef Name,
                                             ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_NONDETERMINISTIC_POINTER_ITERATION_ORDER_CHECK_H
