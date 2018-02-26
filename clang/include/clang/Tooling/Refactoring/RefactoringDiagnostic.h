//===--- RefactoringDiagnostic.h - ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_REFACTORING_REFACTORINGDIAGNOSTIC_H
#define LLVM_CLANG_TOOLING_REFACTORING_REFACTORINGDIAGNOSTIC_H

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/PartialDiagnostic.h"

namespace clang {
namespace diag {
enum {
#define DIAG(ENUM, FLAGS, DEFAULT_MAPPING, DESC, GROUP, SFINAE, NOWERROR,      \
             SHOWINSYSHEADER, CATEGORY)                                        \
  ENUM,
#define REFACTORINGSTART
#include "clang/Basic/DiagnosticRefactoringKinds.inc"
#undef DIAG
  NUM_BUILTIN_REFACTORING_DIAGNOSTICS
};
} // end namespace diag
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_REFACTORING_REFACTORINGDIAGNOSTIC_H
