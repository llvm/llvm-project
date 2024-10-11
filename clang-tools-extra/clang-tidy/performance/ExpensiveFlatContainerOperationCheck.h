//===--- ExpensiveFlatContainerOperationCheck.h - clang-tidy ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_EXPENSIVEFLATCONTAINEROPERATIONCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_EXPENSIVEFLATCONTAINEROPERATIONCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::performance {

/// Warns when calling an O(N) operation on a flat container.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/performance/expensive-flat-container-operation.html
class ExpensiveFlatContainerOperationCheck : public ClangTidyCheck {
public:
  ExpensiveFlatContainerOperationCheck(StringRef Name,
                                       ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override;
  std::optional<TraversalKind> getCheckTraversalKind() const override;

private:
  bool WarnOutsideLoops;
  std::vector<StringRef> FlatContainers;
};

} // namespace clang::tidy::performance

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_EXPENSIVEFLATCONTAINEROPERATIONCHECK_H
