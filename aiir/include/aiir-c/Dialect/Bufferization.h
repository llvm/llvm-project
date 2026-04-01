//===-- aiir-c/Dialect/Bufferization.h - C API for Bufferization Dialect
//--------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_C_DIALECT_BUFFERIZATION_H
#define AIIR_C_DIALECT_BUFFERIZATION_H

#include "aiir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

AIIR_DECLARE_CAPI_DIALECT_REGISTRATION(Bufferization, bufferization);

#ifdef __cplusplus
}
#endif

#include "aiir/Dialect/Bufferization/Transforms/Passes.capi.h.inc"

#endif // AIIR_C_DIALECT_BUFFERIZATION_H
