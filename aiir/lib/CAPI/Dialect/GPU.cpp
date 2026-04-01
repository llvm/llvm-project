//===- GPU.cpp - C Interface for GPU dialect ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Dialect/GPU.h"
#include "aiir/CAPI/Registration.h"
#include "aiir/Dialect/GPU/IR/GPUDialect.h"
#include "llvm/Support/Casting.h"

using namespace aiir;

AIIR_DEFINE_CAPI_DIALECT_REGISTRATION(GPU, gpu, gpu::GPUDialect)

//===-------------------------------------------------------------------===//
// AsyncTokenType
//===-------------------------------------------------------------------===//

bool aiirTypeIsAGPUAsyncTokenType(AiirType type) {
  return isa<gpu::AsyncTokenType>(unwrap(type));
}

AiirType aiirGPUAsyncTokenTypeGet(AiirContext ctx) {
  return wrap(gpu::AsyncTokenType::get(unwrap(ctx)));
}

AiirStringRef aiirGPUAsyncTokenTypeGetName(void) {
  return wrap(gpu::AsyncTokenType::name);
}

//===---------------------------------------------------------------------===//
// ObjectAttr
//===---------------------------------------------------------------------===//

bool aiirAttributeIsAGPUObjectAttr(AiirAttribute attr) {
  return llvm::isa<gpu::ObjectAttr>(unwrap(attr));
}

AiirAttribute aiirGPUObjectAttrGet(AiirContext aiirCtx, AiirAttribute target,
                                   uint32_t format, AiirStringRef objectStrRef,
                                   AiirAttribute aiirObjectProps) {
  AIIRContext *ctx = unwrap(aiirCtx);
  llvm::StringRef object = unwrap(objectStrRef);
  DictionaryAttr objectProps;
  if (aiirObjectProps.ptr != nullptr)
    objectProps = llvm::cast<DictionaryAttr>(unwrap(aiirObjectProps));
  return wrap(gpu::ObjectAttr::get(
      ctx, unwrap(target), static_cast<gpu::CompilationTarget>(format),
      StringAttr::get(ctx, object), objectProps, nullptr));
}

AiirStringRef aiirGPUObjectAttrGetName(void) {
  return wrap(gpu::ObjectAttr::name);
}

AiirAttribute aiirGPUObjectAttrGetWithKernels(AiirContext aiirCtx,
                                              AiirAttribute target,
                                              uint32_t format,
                                              AiirStringRef objectStrRef,
                                              AiirAttribute aiirObjectProps,
                                              AiirAttribute aiirKernelsAttr) {
  AIIRContext *ctx = unwrap(aiirCtx);
  llvm::StringRef object = unwrap(objectStrRef);
  DictionaryAttr objectProps;
  if (aiirObjectProps.ptr != nullptr)
    objectProps = llvm::cast<DictionaryAttr>(unwrap(aiirObjectProps));
  gpu::KernelTableAttr kernels;
  if (aiirKernelsAttr.ptr != nullptr)
    kernels = llvm::cast<gpu::KernelTableAttr>(unwrap(aiirKernelsAttr));
  return wrap(gpu::ObjectAttr::get(
      ctx, unwrap(target), static_cast<gpu::CompilationTarget>(format),
      StringAttr::get(ctx, object), objectProps, kernels));
}

AiirAttribute aiirGPUObjectAttrGetTarget(AiirAttribute aiirObjectAttr) {
  gpu::ObjectAttr objectAttr =
      llvm::cast<gpu::ObjectAttr>(unwrap(aiirObjectAttr));
  return wrap(objectAttr.getTarget());
}

uint32_t aiirGPUObjectAttrGetFormat(AiirAttribute aiirObjectAttr) {
  gpu::ObjectAttr objectAttr =
      llvm::cast<gpu::ObjectAttr>(unwrap(aiirObjectAttr));
  return static_cast<uint32_t>(objectAttr.getFormat());
}

AiirStringRef aiirGPUObjectAttrGetObject(AiirAttribute aiirObjectAttr) {
  gpu::ObjectAttr objectAttr =
      llvm::cast<gpu::ObjectAttr>(unwrap(aiirObjectAttr));
  llvm::StringRef object = objectAttr.getObject();
  return aiirStringRefCreate(object.data(), object.size());
}

bool aiirGPUObjectAttrHasProperties(AiirAttribute aiirObjectAttr) {
  gpu::ObjectAttr objectAttr =
      llvm::cast<gpu::ObjectAttr>(unwrap(aiirObjectAttr));
  return objectAttr.getProperties() != nullptr;
}

AiirAttribute aiirGPUObjectAttrGetProperties(AiirAttribute aiirObjectAttr) {
  gpu::ObjectAttr objectAttr =
      llvm::cast<gpu::ObjectAttr>(unwrap(aiirObjectAttr));
  return wrap(objectAttr.getProperties());
}

bool aiirGPUObjectAttrHasKernels(AiirAttribute aiirObjectAttr) {
  gpu::ObjectAttr objectAttr =
      llvm::cast<gpu::ObjectAttr>(unwrap(aiirObjectAttr));
  return objectAttr.getKernels() != nullptr;
}

AiirAttribute aiirGPUObjectAttrGetKernels(AiirAttribute aiirObjectAttr) {
  gpu::ObjectAttr objectAttr =
      llvm::cast<gpu::ObjectAttr>(unwrap(aiirObjectAttr));
  return wrap(objectAttr.getKernels());
}
