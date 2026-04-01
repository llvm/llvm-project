//===-- aiir-c/Dialect/Async.h - C API for Async dialect ---------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#ifndef AIIR_C_DIALECT_ASYNC_H
#define AIIR_C_DIALECT_ASYNC_H

#include "aiir-c/IR.h"
#include "aiir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

AIIR_DECLARE_CAPI_DIALECT_REGISTRATION(Async, async);

#ifdef __cplusplus
}
#endif

#include "aiir/Dialect/Async/Passes.capi.h.inc"

#endif // AIIR_C_DIALECT_ASYNC_H
