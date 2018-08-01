//===--- RefactoringActionFinder.cpp - Clang refactoring library ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Implements methods that find the refactoring actions that can be
/// performed at specific locations / source ranges in a translation unit.
///
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Refactor/RefactoringActionFinder.h"
#include "clang/Tooling/Refactor/RefactoringOperation.h"
#include "clang/Tooling/Refactor/USRFinder.h"

namespace clang {
namespace tooling {

RefactoringActionSet findActionSetAt(SourceLocation Location,
                                     SourceRange SelectionRange,
                                     ASTContext &Context) {
  RefactoringActionSet Result;
  if (const auto *ND = rename::getNamedDeclAt(Context, Location))
    Result.Actions.push_back(isLocalSymbol(ND, Context.getLangOpts())
                                 ? RefactoringActionType::Rename_Local
                                 : RefactoringActionType::Rename);

  // FIXME: We can avoid checking if some actions can be initiated when they're
  // not allowed in the current language mode.
  RefactoringActionType Actions[] = {
#define REFACTORING_OPERATION_ACTION(Name, Spelling, Command)                  \
  RefactoringActionType::Name,
#include "clang/Tooling/Refactor/RefactoringActions.def"
  };

  for (auto Action : Actions) {
    auto Op = initiateRefactoringOperationAt(Location, SelectionRange, Context,
                                             Action,
                                             /*CreateOperation=*/true);
    if (Op.Initiated) {
      Result.Actions.push_back(Action);
      if (Op.RefactoringOp) {
        for (const auto &SubAction : Op.RefactoringOp->getAvailableSubActions())
          Result.Actions.push_back(SubAction);
      }
    }
  }

  return Result;
}

} // end namespace tooling
} // end namespace clang
