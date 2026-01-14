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
#include "mlir/IR/PatternMatch.h"
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
    llvm_unreachable("Windows ABI NYI");
  }

  llvm_unreachable("invalid C++ ABI kind");
}

static std::unique_ptr<TargetLoweringInfo>
createTargetLoweringInfo(LowerModule &lm) {
  assert(!cir::MissingFeatures::targetLoweringInfo());
  return std::make_unique<TargetLoweringInfo>();
}

LowerModule::LowerModule(clang::LangOptions langOpts,
                         clang::CodeGenOptions codeGenOpts,
                         mlir::ModuleOp &module,
                         std::unique_ptr<clang::TargetInfo> target,
                         mlir::PatternRewriter &rewriter)
    : module(module), target(std::move(target)), abi(createCXXABI(*this)),
      rewriter(rewriter) {}

const TargetLoweringInfo &LowerModule::getTargetLoweringInfo() {
  if (!targetLoweringInfo)
    targetLoweringInfo = createTargetLoweringInfo(*this);
  return *targetLoweringInfo;
}

// TODO: not to create it every time
std::unique_ptr<LowerModule>
createLowerModule(mlir::ModuleOp module, mlir::PatternRewriter &rewriter) {
  // Fetch target information.
  llvm::Triple triple(mlir::cast<mlir::StringAttr>(
                          module->getAttr(cir::CIRDialect::getTripleAttrName()))
                          .getValue());
  clang::TargetOptions targetOptions;
  targetOptions.Triple = triple.str();
  auto targetInfo = clang::targets::AllocateTarget(triple, targetOptions);

  // FIXME(cir): This just uses the default language options. We need to account
  // for custom options.
  // Create context.
  assert(!cir::MissingFeatures::lowerModuleLangOpts());
  clang::LangOptions langOpts;

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
                                       std::move(targetInfo), rewriter);
}

} // namespace cir
