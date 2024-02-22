//===--- RefactoringActions.cpp - Clang refactoring library ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Contains a list of all the supported refactoring actions.
///
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Refactor/RefactoringActions.h"
#include "llvm/Support/ErrorHandling.h"

namespace clang {
namespace tooling {

StringRef getRefactoringActionTypeName(RefactoringActionType Action) {
  switch (Action) {
#define REFACTORING_ACTION(Name, Spelling)                                     \
  case RefactoringActionType::Name:                                            \
    return Spelling;
#include "clang/Tooling/Refactor/RefactoringActions.def"
  }
  llvm_unreachable("unexpected RefactoringActionType value");
}

} // end namespace tooling
} // end namespace clang
