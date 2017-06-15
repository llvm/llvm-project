//===--- RefactoringActions.cpp - Clang refactoring library ---------------===//
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

#include "clang/Tooling/Refactor/RefactoringActions.h"

namespace clang {
namespace tooling {

StringRef getRefactoringActionTypeName(RefactoringActionType Action) {
  switch (Action) {
#define REFACTORING_ACTION(Name, Spelling)                                     \
  case RefactoringActionType::Name:                                            \
    return Spelling;
#include "clang/Tooling/Refactor/RefactoringActions.def"
  }
}

} // end namespace tooling
} // end namespace clang
