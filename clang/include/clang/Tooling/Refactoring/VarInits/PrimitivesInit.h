//===--- PrimitivesInit.h - Clang refactoring library ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_REFACTORING_VARINITS_PRIMITIVES_INIT_H
#define LLVM_CLANG_TOOLING_REFACTORING_VARINITS_PRIMITIVES_INIT_H

#include "clang/Tooling/Refactoring/RefactoringActionRules.h"

namespace clang {
namespace tooling {

/// \c PrimitivesInit performs the initialization of
/// a selected primitive variable with a default value
class PrimitivesInit final : public SourceChangeRefactoringRule {
public:
  /// \param Variable        declaration-only VarDecl
  static Expected<PrimitivesInit>
  initiate(RefactoringRuleContext &Context, VarDecl *Variable);

  static const RefactoringDescriptor &describe();

private:
  PrimitivesInit(VarDecl *Variable)
      : Variable(std::move(Variable)) {}

  Expected<AtomicChanges>
  createSourceReplacements(RefactoringRuleContext &Context) override;

  VarDecl *Variable;
};

} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_REFACTORING_VARINITS_PRIMITIVES_INIT_H
