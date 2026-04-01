//===-- aiir-c/Dialect/Ptr.h - C API for Ptr Dialect --------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_C_DIALECT_PTR_H
#define AIIR_C_DIALECT_PTR_H

#include "aiir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

AIIR_DECLARE_CAPI_DIALECT_REGISTRATION(Ptr, ptr);

#ifdef __cplusplus
}
#endif

#endif // AIIR_C_DIALECT_PTR_H
