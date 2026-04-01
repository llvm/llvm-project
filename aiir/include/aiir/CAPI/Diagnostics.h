//===- IR.h - C API Utils for AIIR Diagnostics ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CAPI_DIAGNOSTICS_H
#define AIIR_CAPI_DIAGNOSTICS_H

#include "aiir-c/Diagnostics.h"
#include <cassert>

namespace aiir {
class Diagnostic;
} // namespace aiir

inline aiir::Diagnostic &unwrap(AiirDiagnostic diagnostic) {
  assert(diagnostic.ptr && "unexpected null diagnostic");
  return *(static_cast<aiir::Diagnostic *>(diagnostic.ptr));
}

inline AiirDiagnostic wrap(aiir::Diagnostic &diagnostic) {
  return {&diagnostic};
}

#endif // AIIR_CAPI_DIAGNOSTICS_H
