//===--- RefactoringActionRule.h - Clang refactoring library -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_REFACTOR_REFACTORING_ACTION_RULE_H
#define LLVM_CLANG_TOOLING_REFACTOR_REFACTORING_ACTION_RULE_H

#include "clang/Basic/LLVM.h"
#include "clang/Tooling/Refactoring/AtomicChange.h"
#include "llvm/Support/Error.h"
#include <vector>

namespace clang {
namespace tooling {

class RefactoringRuleContext;

/// A common refactoring action rule interface.
class RefactoringActionRule {
public:
  enum RuleKind { SourceChangeRefactoringRuleKind };

  RuleKind getRuleKind() const { return Kind; }

  virtual ~RefactoringActionRule() {}

protected:
  RefactoringActionRule(RuleKind Kind) : Kind(Kind) {}

private:
  RuleKind Kind;
};

/// A type of refactoring action rule that produces source replacements in the
/// form of atomic changes.
///
/// This action rule is typically used for local refactorings that replace
/// source in a single AST unit.
class SourceChangeRefactoringRule : public RefactoringActionRule {
public:
  SourceChangeRefactoringRule()
      : RefactoringActionRule(SourceChangeRefactoringRuleKind) {}

  /// Initiates and performs a refactoring action that modifies the sources.
  ///
  /// The specific rule must return an llvm::Error with a DiagnosticError
  /// payload or None when the refactoring action couldn't be initiated/
  /// performed, or \c AtomicChanges when the action was performed successfully.
  virtual Expected<Optional<AtomicChanges>>
  createSourceReplacements(RefactoringRuleContext &Context) = 0;

  static bool classof(const RefactoringActionRule *Rule) {
    return Rule->getRuleKind() == SourceChangeRefactoringRuleKind;
  }
};

/// A set of refactoring action rules that should have unique initiation
/// requirements.
using RefactoringActionRules =
    std::vector<std::unique_ptr<RefactoringActionRule>>;

} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_REFACTOR_REFACTORING_ACTION_RULE_H
