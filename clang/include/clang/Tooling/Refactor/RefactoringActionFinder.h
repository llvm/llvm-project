//===--- RefactoringActionFinder.h - Clang refactoring library ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Provides methods to find the refactoring actions that can be
/// performed at specific locations / source ranges in a translation unit.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_REFACTOR_REFACTORING_ACTION_FINDER_H
#define LLVM_CLANG_TOOLING_REFACTOR_REFACTORING_ACTION_FINDER_H

#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Tooling/Refactor/RefactoringActions.h"
#include "llvm/ADT/StringSet.h"
#include <vector>

namespace clang {

class NamedDecl;
class ASTContext;

namespace tooling {

/// Contains a set of a refactoring actions.
struct RefactoringActionSet {
  /// A set of refactoring actions that can be performed at some specific
  /// location in a source file.
  ///
  /// The actions in the action set are ordered by their priority: most
  /// important actions are placed before the less important ones.
  std::vector<RefactoringActionType> Actions;

  RefactoringActionSet() {}

  RefactoringActionSet(RefactoringActionSet &&) = default;
  RefactoringActionSet &operator=(RefactoringActionSet &&) = default;
};

/// \brief Returns a \c RefactoringActionSet that contains the set of actions
/// that can be performed at the given location.
RefactoringActionSet findActionSetAt(SourceLocation Loc,
                                     SourceRange SelectionRange,
                                     ASTContext &Context);

/// \brief Returns a set of USRs that correspond to the given declaration.
llvm::StringSet<> findSymbolsUSRSet(const NamedDecl *FoundDecl,
                                    ASTContext &Context);

} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_REFACTOR_REFACTORING_ACTION_FINDER_H
