//===--- DeleteStatementRule.h - Clang refactoring library ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_REFACTORING_DELETE_DELETESTATEMENTRULE_H
#define LLVM_CLANG_TOOLING_REFACTORING_DELETE_DELETESTATEMENTRULE_H

#include "clang/Tooling/Refactoring/RefactoringActionRules.h"

namespace clang {
namespace tooling {

/// A "Delete Statement" refactoring rule deletes code around given statement
class DeleteStatementRule final : public SourceChangeRefactoringRule {
public:
  /// Initiates the delete statement refactoring operation.
  ///
  /// \param Statement    Statement to delete.
  static Expected<DeleteStatementRule>
  initiate(RefactoringRuleContext &Context, Stmt *Stmt);

  static const RefactoringDescriptor &describe();

private:
  DeleteStatementRule(Stmt *Stmt)
      : Statement(std::move(Stmt)) {}

  Expected<AtomicChanges>
  createSourceReplacements(RefactoringRuleContext &Context) override;

  Stmt *Statement;
};

} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_REFACTORING_DELETE_DELETESTATEMENTRULE_H
