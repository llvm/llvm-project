//===- AMDGPU.cpp - C Interface for AMDGPU dialect ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Dialect/AMDGPU.h"
#include "aiir/CAPI/Registration.h"
#include "aiir/Dialect/AMDGPU/IR/AMDGPUDialect.h"

AIIR_DEFINE_CAPI_DIALECT_REGISTRATION(AMDGPU, amdgpu,
                                      aiir::amdgpu::AMDGPUDialect)

using namespace aiir;
using namespace aiir::amdgpu;

//===---------------------------------------------------------------------===//
// TDMBaseType
//===---------------------------------------------------------------------===//

bool aiirTypeIsAAMDGPUTDMBaseType(AiirType type) {
  return isa<amdgpu::TDMBaseType>(unwrap(type));
}

AiirTypeID aiirAMDGPUTDMBaseTypeGetTypeID() {
  return wrap(amdgpu::TDMBaseType::getTypeID());
}

AiirType aiirAMDGPUTDMBaseTypeGet(AiirContext ctx, AiirType elementType) {
  return wrap(amdgpu::TDMBaseType::get(unwrap(ctx), unwrap(elementType)));
}

AiirStringRef aiirAMDGPUTDMBaseTypeGetName(void) {
  return wrap(amdgpu::TDMBaseType::name);
}

//===---------------------------------------------------------------------===//
// TDMDescriptorType
//===---------------------------------------------------------------------===//

bool aiirTypeIsAAMDGPUTDMDescriptorType(AiirType type) {
  return isa<amdgpu::TDMDescriptorType>(unwrap(type));
}

AiirTypeID aiirAMDGPUTDMDescriptorTypeGetTypeID() {
  return wrap(amdgpu::TDMDescriptorType::getTypeID());
}

AiirType aiirAMDGPUTDMDescriptorTypeGet(AiirContext ctx) {
  return wrap(amdgpu::TDMDescriptorType::get(unwrap(ctx)));
}

AiirStringRef aiirAMDGPUTDMDescriptorTypeGetName(void) {
  return wrap(amdgpu::TDMDescriptorType::name);
}

//===---------------------------------------------------------------------===//
// TDMGatherBaseType
//===---------------------------------------------------------------------===//

bool aiirTypeIsAAMDGPUTDMGatherBaseType(AiirType type) {
  return isa<amdgpu::TDMGatherBaseType>(unwrap(type));
}

AiirTypeID aiirAMDGPUTDMGatherBaseTypeGetTypeID() {
  return wrap(amdgpu::TDMGatherBaseType::getTypeID());
}

AiirType aiirAMDGPUTDMGatherBaseTypeGet(AiirContext ctx, AiirType elementType,
                                        AiirType indexType) {
  return wrap(amdgpu::TDMGatherBaseType::get(unwrap(ctx), unwrap(elementType),
                                             unwrap(indexType)));
}

AiirStringRef aiirAMDGPUTDMGatherBaseTypeGetName(void) {
  return wrap(amdgpu::TDMGatherBaseType::name);
}
