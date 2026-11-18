//===--- SwitchToIf.h - Switch to if refactoring -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_REFACTORING_SWITCHTOIF_SWITCHTOIF_H
#define LLVM_CLANG_TOOLING_REFACTORING_SWITCHTOIF_SWITCHTOIF_H

#include "clang/Tooling/Refactoring/ASTSelection.h"
#include "clang/Tooling/Refactoring/RefactoringActionRules.h"

namespace clang {
class SwitchStmt;

namespace tooling {

/// A "Switch to If" refactoring converts a switch statement into an if-else
/// chain.
class SwitchToIf final : public SourceChangeRefactoringRule {
public:
  /// Initiates the switch-to-if refactoring operation.
  ///
  /// \param Selection The selected AST node, which should be a switch statement.
  static Expected<SwitchToIf>
  initiate(RefactoringRuleContext &Context,
           SelectedASTNode Selection);

  static const RefactoringDescriptor &describe();

private:
  SwitchToIf(const SwitchStmt *Switch) : TheSwitch(Switch) {}

  Expected<AtomicChanges>
  createSourceReplacements(RefactoringRuleContext &Context) override;

  const SwitchStmt *TheSwitch;
};

} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_REFACTORING_SWITCHTOIF_SWITCHTOIF_H

