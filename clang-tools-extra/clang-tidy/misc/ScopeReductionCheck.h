//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file
/// This file defines ScopeReductionCheck, a clang-tidy checker that identifies
/// variables that can be declared in smaller scopes to improve code locality
/// and readability.

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_SCOPEREDUCTIONCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_SCOPEREDUCTIONCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::misc {

/// Detects variables that can be declared in smaller scopes.
///
/// This checker analyzes variable declarations and their usage patterns to
/// determine if they can be moved to a more restrictive scope, improving
/// code locality and reducing the variable's lifetime.
///
/// The checker uses a 7-step algorithm to perform scope analysis and only
/// reports cases where variables can be moved to genuinely smaller scopes.
class ScopeReductionCheck : public ClangTidyCheck {
public:
  ScopeReductionCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  /// Find the first ancestor of the specified type by walking up the AST.
  /// Returns nullptr if no ancestor of the specified type is found.
  template <typename T>
  const T *findAncestorOfType(const Stmt *Start, ParentMapContext &Parents);

  /// Check if a statement is inside any type of loop.
  bool isInLoop(const Stmt *Use, ParentMapContext &Parents);

  /// Find the containing switch case for a statement.
  /// Returns nullptr if not inside a switch case.
  const clang::SwitchCase *findContainingSwitchCase(const Stmt *Use,
                                                    ParentMapContext &Parents);

  /// Emit diagnostic notes showing where the variable is used.
  /// Limits output to avoid excessive noise in diagnostics.
  void emitUsageNotes(const llvm::SmallVector<const DeclRefExpr *, 8> &Uses);

  /// Utility function to take the first N elements from a container.
  /// Used to limit the number of usage notes displayed.
  template <typename Container>
  auto take(const Container &ThisContainer, size_t Count) {
    return llvm::make_range(ThisContainer.begin(),
                            ThisContainer.begin() +
                                std::min(Count, ThisContainer.size()));
  }
};

} // namespace clang::tidy::misc

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_SCOPEREDUCTIONCHECK_H
