//===-- aiir-c/Dialect/Arith.h - C API for Arith dialect ----------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface for registering and accessing the
// Arith dialect. A dialect should be registered with a context to make it
// available to users of the context. These users must load the dialect
// before using any of its attributes, operations or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_C_DIALECT_ARITH_H
#define AIIR_C_DIALECT_ARITH_H

#include "aiir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

AIIR_DECLARE_CAPI_DIALECT_REGISTRATION(Arith, arith);

#ifdef __cplusplus
}
#endif

#include "aiir/Dialect/Arith/Transforms/Passes.capi.h.inc"

#endif // AIIR_C_DIALECT_ARITH_H
