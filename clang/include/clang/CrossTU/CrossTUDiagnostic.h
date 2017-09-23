//===--- CrossTUDiagnostic.h - Diagnostics for Cross TU ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CROSSTU_CROSSTUDIAGNOSTIC_H
#define LLVM_CLANG_CROSSTU_CROSSTUDIAGNOSTIC_H

#include "clang/Basic/Diagnostic.h"

namespace clang {
namespace diag {
enum {
#define DIAG(ENUM, FLAGS, DEFAULT_MAPPING, DESC, GROUP, SFINAE, NOWERROR,      \
             SHOWINSYSHEADER, CATEGORY)                                        \
  ENUM,
#define CROSSTUSTART
#include "clang/Basic/DiagnosticCrossTUKinds.inc"
#undef DIAG
  NUM_BUILTIN_CROSSTU_DIAGNOSTICS
};
} // end namespace diag
} // end namespace clang

#endif // LLVM_CLANG_FRONTEND_FRONTENDDIAGNOSTIC_H
