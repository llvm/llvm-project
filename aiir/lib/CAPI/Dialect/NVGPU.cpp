//===- NVGPU.cpp - C Interface for NVGPU dialect ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Dialect/NVGPU.h"
#include "aiir/CAPI/Registration.h"
#include "aiir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "aiir/IR/BuiltinTypes.h"

using namespace aiir;
using namespace aiir::nvgpu;

AIIR_DEFINE_CAPI_DIALECT_REGISTRATION(NVGPU, nvgpu, aiir::nvgpu::NVGPUDialect)

bool aiirTypeIsANVGPUTensorMapDescriptorType(AiirType type) {
  return isa<nvgpu::TensorMapDescriptorType>(unwrap(type));
}

AiirType aiirNVGPUTensorMapDescriptorTypeGet(AiirContext ctx,
                                             AiirType tensorMemrefType,
                                             int swizzle, int l2promo,
                                             int oobFill, int interleave) {
  return wrap(nvgpu::TensorMapDescriptorType::get(
      unwrap(ctx), cast<MemRefType>(unwrap(tensorMemrefType)),
      TensorMapSwizzleKind(swizzle), TensorMapL2PromoKind(l2promo),
      TensorMapOOBKind(oobFill), TensorMapInterleaveKind(interleave)));
}

AiirStringRef aiirNVGPUTensorMapDescriptorTypeGetName(void) {
  return wrap(nvgpu::TensorMapDescriptorType::name);
}
