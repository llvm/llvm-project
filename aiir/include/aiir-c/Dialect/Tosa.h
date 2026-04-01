//===-- aiir-c/Dialect/Tosa.h - C API for Tosa Dialect --------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_C_DIALECT_TOSA_H
#define AIIR_C_DIALECT_TOSA_H

#include "aiir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

AIIR_DECLARE_CAPI_DIALECT_REGISTRATION(Tosa, tosa);

#ifdef __cplusplus
}
#endif

#include "aiir/Dialect/Tosa/Transforms/Passes.capi.h.inc"

#endif // AIIR_C_DIALECT_TOSA_H
