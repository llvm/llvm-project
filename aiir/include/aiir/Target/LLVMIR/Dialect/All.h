//===- All.h - AIIR To LLVM IR Translation Registration ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to register the translations of all suitable
// dialects to LLVM IR.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TARGET_LLVMIR_DIALECT_ALL_H
#define AIIR_TARGET_LLVMIR_DIALECT_ALL_H

#include "aiir/Target/LLVMIR/Dialect/ArmNeon/ArmNeonToLLVMIRTranslation.h"
#include "aiir/Target/LLVMIR/Dialect/ArmSME/ArmSMEToLLVMIRTranslation.h"
#include "aiir/Target/LLVMIR/Dialect/ArmSVE/ArmSVEToLLVMIRTranslation.h"
#include "aiir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "aiir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "aiir/Target/LLVMIR/Dialect/LLVMIR/LLVMIRToLLVMTranslation.h"
#include "aiir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "aiir/Target/LLVMIR/Dialect/NVVM/LLVMIRToNVVMTranslation.h"
#include "aiir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "aiir/Target/LLVMIR/Dialect/OpenACC/OpenACCToLLVMIRTranslation.h"
#include "aiir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "aiir/Target/LLVMIR/Dialect/Ptr/PtrToLLVMIRTranslation.h"
#include "aiir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "aiir/Target/LLVMIR/Dialect/SPIRV/SPIRVToLLVMIRTranslation.h"
#include "aiir/Target/LLVMIR/Dialect/VCIX/VCIXToLLVMIRTranslation.h"
#include "aiir/Target/LLVMIR/Dialect/XeVM/XeVMToLLVMIRTranslation.h"

namespace aiir {
class DialectRegistry;

/// Registers all dialects that can be translated to LLVM IR and the
/// corresponding translation interfaces.
static inline void registerAllToLLVMIRTranslations(DialectRegistry &registry) {
  registerArmNeonDialectTranslation(registry);
  registerArmSMEDialectTranslation(registry);
  registerArmSVEDialectTranslation(registry);
  registerBuiltinDialectTranslation(registry);
  registerGPUDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  registerNVVMDialectTranslation(registry);
  registerOpenACCDialectTranslation(registry);
  registerOpenMPDialectTranslation(registry);
  registerPtrDialectTranslation(registry);
  registerROCDLDialectTranslation(registry);
  registerSPIRVDialectTranslation(registry);
  registerVCIXDialectTranslation(registry);
  registerXeVMDialectTranslation(registry);

  // Extension required for translating GPU offloading Ops.
  gpu::registerOffloadingLLVMTranslationInterfaceExternalModels(registry);
}

/// Registers all the translations to LLVM IR required by GPU passes.
/// TODO: Remove this function when a safe dialect interface registration
/// mechanism is implemented, see D157703.
static inline void
registerAllGPUToLLVMIRTranslations(DialectRegistry &registry) {
  registerBuiltinDialectTranslation(registry);
  registerGPUDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  registerNVVMDialectTranslation(registry);
  registerROCDLDialectTranslation(registry);
  registerSPIRVDialectTranslation(registry);
  registerXeVMDialectTranslation(registry);

  // Extension required for translating GPU offloading Ops.
  gpu::registerOffloadingLLVMTranslationInterfaceExternalModels(registry);
}

/// Registers all dialects that can be translated from LLVM IR and the
/// corresponding translation interfaces.
static inline void
registerAllFromLLVMIRTranslations(DialectRegistry &registry) {
  registerLLVMDialectImport(registry);
  registerNVVMDialectImport(registry);
}
} // namespace aiir

#endif // AIIR_TARGET_LLVMIR_DIALECT_ALL_H
