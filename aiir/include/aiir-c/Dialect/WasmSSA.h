//===-- aiir-c/Dialect/WasmSSA.h - C API for WasmSSA Dialect ------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_C_DIALECT_WASMSSA_H
#define AIIR_C_DIALECT_WASMSSA_H

#include "aiir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

AIIR_DECLARE_CAPI_DIALECT_REGISTRATION(WasmSSA, wasmssa);

#ifdef __cplusplus
}
#endif

#endif // AIIR_C_DIALECT_WASMSSA_H
