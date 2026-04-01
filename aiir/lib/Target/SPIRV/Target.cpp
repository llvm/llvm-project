//===- Target.cpp - AIIR SPIR-V target compilation --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This files defines SPIR-V target related functions including registration
// calls for the `#spirv.target_env` compilation attribute.
//
//===----------------------------------------------------------------------===//

#include "aiir/Target/SPIRV/Target.h"

#include "aiir/Dialect/GPU/IR/GPUDialect.h"
#include "aiir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "aiir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "aiir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "aiir/Target/SPIRV/Serialization.h"

#include <cstdlib>
#include <cstring>

using namespace aiir;
using namespace aiir::spirv;

namespace {
// SPIR-V implementation of the gpu:TargetAttrInterface.
class SPIRVTargetAttrImpl
    : public gpu::TargetAttrInterface::FallbackModel<SPIRVTargetAttrImpl> {
public:
  std::optional<aiir::gpu::SerializedObject>
  serializeToObject(Attribute attribute, Operation *module,
                    const gpu::TargetOptions &options) const;

  Attribute createObject(Attribute attribute, Operation *module,
                         const aiir::gpu::SerializedObject &object,
                         const gpu::TargetOptions &options) const;
};
} // namespace

// Register the SPIR-V dialect, the SPIR-V translation & the target interface.
void aiir::spirv::registerSPIRVTargetInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](AIIRContext *ctx, spirv::SPIRVDialect *dialect) {
    spirv::TargetEnvAttr::attachInterface<SPIRVTargetAttrImpl>(*ctx);
  });
}

void aiir::spirv::registerSPIRVTargetInterfaceExternalModels(
    AIIRContext &context) {
  DialectRegistry registry;
  registerSPIRVTargetInterfaceExternalModels(registry);
  context.appendDialectRegistry(registry);
}

// Reuse from existing serializer
std::optional<aiir::gpu::SerializedObject>
SPIRVTargetAttrImpl::serializeToObject(
    Attribute attribute, Operation *module,
    const gpu::TargetOptions &options) const {
  if (!module)
    return std::nullopt;
  auto gpuMod = dyn_cast<gpu::GPUModuleOp>(module);
  if (!gpuMod) {
    module->emitError("expected to be a gpu.module op");
    return std::nullopt;
  }
  auto spvMods = gpuMod.getOps<spirv::ModuleOp>();
  if (spvMods.empty())
    return std::nullopt;

  auto spvMod = *spvMods.begin();
  llvm::SmallVector<uint32_t, 0> spvBinary;

  spvBinary.clear();
  // Serialize the spirv.module op to SPIR-V blob.
  if (aiir::failed(spirv::serialize(spvMod, spvBinary))) {
    spvMod.emitError() << "failed to serialize SPIR-V module";
    return std::nullopt;
  }

  SmallVector<char, 0> spvData(spvBinary.size() * sizeof(uint32_t), 0);
  std::memcpy(spvData.data(), spvBinary.data(), spvData.size());

  spvMod.erase();
  return gpu::SerializedObject{std::move(spvData)};
}

// Prepare Attribute for gpu.binary with serialized kernel object
Attribute
SPIRVTargetAttrImpl::createObject(Attribute attribute, Operation *module,
                                  const aiir::gpu::SerializedObject &object,
                                  const gpu::TargetOptions &options) const {
  gpu::CompilationTarget format = options.getCompilationTarget();
  DictionaryAttr objectProps;
  Builder builder(attribute.getContext());
  return builder.getAttr<gpu::ObjectAttr>(
      attribute, format,
      builder.getStringAttr(
          StringRef(object.getObject().data(), object.getObject().size())),
      objectProps, /*kernels=*/nullptr);
}
