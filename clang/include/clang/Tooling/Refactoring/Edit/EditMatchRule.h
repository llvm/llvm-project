//===--- EditMatchRule.h - Clang refactoring library ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_REFACTORING_EDIT_EDITMATCHRULE_H
#define LLVM_CLANG_TOOLING_REFACTORING_EDIT_EDITMATCHRULE_H

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Refactoring/RefactoringActionRules.h"
#include "clang/Tooling/Transformer/RewriteRule.h"

namespace clang {
namespace tooling {

/// A "Edit Match" refactoring rule edits code around matches according to
/// EditGenerator.
class EditMatchRule final : public SourceChangeRefactoringRule {
public:
  /// Initiates the delete match refactoring operation.
  ///
  /// \param R    MatchResult  Match result to edit.
  /// \param EG    EditGenerator  Edit to perform.
  static Expected<EditMatchRule>
  initiate(RefactoringRuleContext &Context,
           ast_matchers::MatchFinder::MatchResult R,
           transformer::EditGenerator EG);

  static const RefactoringDescriptor &describe();

private:
  EditMatchRule(ast_matchers::MatchFinder::MatchResult R,
                transformer::EditGenerator EG)
      : Result(std::move(R)), EditGenerator(std::move(EG)) {}

  Expected<AtomicChanges>
  createSourceReplacements(RefactoringRuleContext &Context) override;

  ast_matchers::MatchFinder::MatchResult Result;
  transformer::EditGenerator EditGenerator;
};

} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_REFACTORING_EDIT_EDITMATCHRULE_H
