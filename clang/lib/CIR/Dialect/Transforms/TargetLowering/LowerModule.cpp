//===--- LowerModule.cpp - Lower CIR Module to a Target -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics clang/lib/CodeGen/CodeGenModule.cpp. The queries
// are adapted to operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//

#include "LowerModule.h"
#include "CIRCXXABI.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"

namespace cir {

static std::unique_ptr<CIRCXXABI> createCXXABI(LowerModule &lm) {
  switch (lm.getCXXABIKind()) {
  case clang::TargetCXXABI::AppleARM64:
  case clang::TargetCXXABI::Fuchsia:
  case clang::TargetCXXABI::GenericAArch64:
  case clang::TargetCXXABI::GenericARM:
  case clang::TargetCXXABI::iOS:
  case clang::TargetCXXABI::WatchOS:
  case clang::TargetCXXABI::GenericMIPS:
  case clang::TargetCXXABI::GenericItanium:
  case clang::TargetCXXABI::WebAssembly:
  case clang::TargetCXXABI::XL:
    return createItaniumCXXABI(lm);
  case clang::TargetCXXABI::Microsoft:
    return createMicrosoftCXXABI(lm);
  }

  llvm_unreachable("invalid C++ ABI kind");
}

static std::unique_ptr<TargetLoweringInfo>
createTargetLoweringInfo(LowerModule &lm) {
  const llvm::Triple &triple = lm.getTarget().getTriple();

  switch (triple.getArch()) {
  case llvm::Triple::amdgpu:
    return createAMDGPUTargetLoweringInfo();
  case llvm::Triple::nvptx:
  case llvm::Triple::nvptx64:
    return createNVPTXTargetLoweringInfo();
  case llvm::Triple::spir:
  case llvm::Triple::spir64:
  case llvm::Triple::spirv:
  case llvm::Triple::spirv32:
  case llvm::Triple::spirv64:
    return createSPIRVTargetLoweringInfo();
  default:
    assert(!cir::MissingFeatures::targetLoweringInfo());
    return std::make_unique<TargetLoweringInfo>();
  }
}

LowerModule::LowerModule(clang::LangOptions langOpts,
                         clang::CodeGenOptions codeGenOpts,
                         mlir::ModuleOp &module,
                         std::unique_ptr<clang::TargetInfo> target)
    : module(module), langOpts(std::move(langOpts)), target(std::move(target)),
      abi(createCXXABI(*this)) {}

const TargetLoweringInfo &LowerModule::getTargetLoweringInfo() {
  if (!targetLoweringInfo)
    targetLoweringInfo = createTargetLoweringInfo(*this);
  return *targetLoweringInfo;
}

// TODO: not to create it every time
std::unique_ptr<LowerModule> createLowerModule(mlir::ModuleOp module) {
  // If the triple is not present, e.g. CIR modules parsed from text, we
  // cannot init LowerModule properly.
  assert(!cir::MissingFeatures::makeTripleAlwaysPresent());
  if (!module->hasAttr(cir::CIRDialect::getTripleAttrName()))
    return nullptr;

  // Fetch target information.
  llvm::Triple triple(mlir::cast<mlir::StringAttr>(
                          module->getAttr(cir::CIRDialect::getTripleAttrName()))
                          .getValue());
  clang::TargetOptions targetOptions;
  targetOptions.Triple = triple.str();
  auto targetInfo = clang::targets::AllocateTarget(triple, targetOptions);

  // Populate the lowering-relevant LangOptions from the module's
  // #cir.lowering_lang_options attribute so a reloaded .cir lowers the same
  // way it was compiled, without a live clang::LangOptions. When the attribute
  // is absent (e.g. hand-written CIR) the defaults are kept;
  clang::LangOptions langOpts;
  if (auto loweringLangOpts =
          mlir::dyn_cast_if_present<cir::LoweringLangOptionsAttr>(
              module->getAttr(
                  cir::CIRDialect::getLoweringLangOptionsAttrName()))) {
    langOpts.Exceptions = loweringLangOpts.getExceptions();
    langOpts.ThreadsafeStatics = loweringLangOpts.getThreadsafeStatics();
    langOpts.CUDA = loweringLangOpts.getCuda();
    langOpts.CUDAIsDevice = loweringLangOpts.getCudaIsDevice();
    langOpts.HIP = loweringLangOpts.getHip();
    langOpts.GPURelocatableDeviceCode = loweringLangOpts.getGpuRdc();
    langOpts.OpenMP = loweringLangOpts.getOpenmp();
    langOpts.OpenMPIsTargetDevice = loweringLangOpts.getOpenmpIsTargetDevice();
    langOpts.setClangABICompat(static_cast<clang::LangOptions::ClangABI>(
        loweringLangOpts.getClangAbiCompat()));
  }

  // FIXME(cir): This just uses the default code generation options. We need to
  // account for custom options.
  assert(!cir::MissingFeatures::lowerModuleCodeGenOpts());
  clang::CodeGenOptions codeGenOpts;

  if (auto optInfo = mlir::cast_if_present<cir::OptInfoAttr>(
          module->getAttr(cir::CIRDialect::getOptInfoAttrName()))) {
    codeGenOpts.OptimizationLevel = optInfo.getLevel();
    codeGenOpts.OptimizeSize = optInfo.getSize();
  }

  return std::make_unique<LowerModule>(std::move(langOpts),
                                       std::move(codeGenOpts), module,
                                       std::move(targetInfo));
}

} // namespace cir
