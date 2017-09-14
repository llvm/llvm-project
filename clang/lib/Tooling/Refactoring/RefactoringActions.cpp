//===--- RefactoringActions.cpp - Constructs refactoring actions ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Refactoring/RefactoringAction.h"

namespace clang {
namespace tooling {

// Forward declare the individual create*Action functions.
#define REFACTORING_ACTION(Name)                                               \
  std::unique_ptr<RefactoringAction> create##Name##Action();
#include "clang/Tooling/Refactoring/RefactoringActionRegistry.def"

std::vector<std::unique_ptr<RefactoringAction>> createRefactoringActions() {
  std::vector<std::unique_ptr<RefactoringAction>> Actions;

#define REFACTORING_ACTION(Name) Actions.push_back(create##Name##Action());
#include "clang/Tooling/Refactoring/RefactoringActionRegistry.def"

  return Actions;
}

RefactoringActionRules RefactoringAction::createActiveActionRules() {
  // FIXME: Filter out rules that are not supported by a particular client.
  return createActionRules();
}

} // end namespace tooling
} // end namespace clang
