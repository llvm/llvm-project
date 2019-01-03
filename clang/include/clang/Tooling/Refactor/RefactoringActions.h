//===--- RefactoringActions.h - Clang refactoring library -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Contains a list of all the supported refactoring actions.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_REFACTOR_REFACTORING_ACTIONS_H
#define LLVM_CLANG_TOOLING_REFACTOR_REFACTORING_ACTIONS_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
namespace tooling {

enum class RefactoringActionType {
#define REFACTORING_ACTION(Name, Spelling) Name,
#include "clang/Tooling/Refactor/RefactoringActions.def"
};

StringRef getRefactoringActionTypeName(RefactoringActionType Action);

} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_REFACTOR_REFACTORING_ACTIONS_H
