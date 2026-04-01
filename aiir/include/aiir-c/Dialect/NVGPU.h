//===-- aiir-c/Dialect/NVGPU.h - C API for NVGPU dialect --*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_C_DIALECT_NVGPU_H
#define AIIR_C_DIALECT_NVGPU_H

#include "aiir-c/IR.h"
#include "aiir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

AIIR_DECLARE_CAPI_DIALECT_REGISTRATION(NVGPU, nvgpu);

//===---------------------------------------------------------------------===//
// TensorMapDescriptorType
//===---------------------------------------------------------------------===//

AIIR_CAPI_EXPORTED bool aiirTypeIsANVGPUTensorMapDescriptorType(AiirType type);

AIIR_CAPI_EXPORTED AiirType aiirNVGPUTensorMapDescriptorTypeGet(
    AiirContext ctx, AiirType tensorMemrefType, int swizzle, int l2promo,
    int oobFill, int interleave);

AIIR_CAPI_EXPORTED AiirStringRef aiirNVGPUTensorMapDescriptorTypeGetName(void);

#ifdef __cplusplus
}
#endif

#include "aiir/Dialect/NVGPU/Transforms/Passes.capi.h.inc"

#endif // AIIR_C_DIALECT_NVGPU_H
