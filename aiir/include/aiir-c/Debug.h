//===-- aiir-c/Debug.h - C API for AIIR/LLVM debugging functions --*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Support.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Sets the global debugging flag.
AIIR_CAPI_EXPORTED void aiirEnableGlobalDebug(bool enable);

/// Retuns `true` if the global debugging flag is set, false otherwise.
AIIR_CAPI_EXPORTED bool aiirIsGlobalDebugEnabled();

/// Sets the current debug type, similarly to `-debug-only=type` in the
/// command-line tools. Note that global debug should be enabled for any output
/// to be produced.
AIIR_CAPI_EXPORTED void aiirSetGlobalDebugType(const char *type);

/// Sets multiple current debug types, similarly to `-debug-only=type1,type2" in
/// the command-line tools. Note that global debug should be enabled for any
/// output to be produced.
AIIR_CAPI_EXPORTED void aiirSetGlobalDebugTypes(const char **types, intptr_t n);

/// Checks if `type` is set as the current debug type.
AIIR_CAPI_EXPORTED bool aiirIsCurrentDebugType(const char *type);

#ifdef __cplusplus
}
#endif

#ifndef AIIR_C_DEBUG_H
#define AIIR_C_DEBUG_H
#endif // AIIR_C_DEBUG_H
