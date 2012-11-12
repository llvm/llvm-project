//===--- DiagnosticDriver.h - Diagnostics for libdriver ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DRIVERDIAGNOSTIC_H
#define LLVM_CLANG_DRIVERDIAGNOSTIC_H

#include "clang/Basic/Diagnostic.h"

namespace clang {
  namespace diag {
    enum {
#define DIAG(ENUM,FLAGS,DEFAULT_MAPPING,DESC,GROUP,\
             SFINAE,ACCESS,NOWERROR,SHOWINSYSHEADER,CATEGORY) ENUM,
#define DRIVERSTART
#include "clang/Basic/DiagnosticDriverKinds.inc"
#undef DIAG
      NUM_BUILTIN_DRIVER_DIAGNOSTICS
    };
  }  // end namespace diag
}  // end namespace clang

#endif
