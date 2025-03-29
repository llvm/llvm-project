//===--- DiagnosticFrontend.h - Diagnostics for frontend --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_DIAGNOSTICFRONTEND_H
#define LLVM_CLANG_BASIC_DIAGNOSTICFRONTEND_H

#include "clang/Basic/Diagnostic.h"

namespace clang {
namespace diag {
enum {
#define DIAG(ENUM, FLAGS, DEFAULT_MAPPING, DESC, GROUP, SFINAE, NOWERROR,      \
             SHOWINSYSHEADER, SHOWINSYSMACRO, DEFERRABLE, CATEGORY)            \
  ENUM,
#define FRONTENDSTART
#include "clang/Basic/DiagnosticFrontendKinds.inc"
#undef DIAG
  NUM_BUILTIN_FRONTEND_DIAGNOSTICS
};

#define DIAG_ENUM(ENUM_NAME)                                                   \
  namespace ENUM_NAME {                                                        \
  enum {
#define DIAG_ENUM_ITEM(IDX, NAME) NAME = IDX,
#define DIAG_ENUM_END()                                                        \
  }                                                                            \
  ;                                                                            \
  }
#include "clang/Basic/DiagnosticFrontendEnums.inc"
#undef DIAG_ENUM_END
#undef DIAG_ENUM_ITEM
#undef DIAG_ENUM
} // end namespace diag
} // end namespace clang

#endif // LLVM_CLANG_BASIC_DIAGNOSTICFRONTEND_H
