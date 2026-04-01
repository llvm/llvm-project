//===-- Optimizer/Support/DataLayout.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Support/DataLayout.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "aiir/Dialect/DLTI/DLTI.h"
#include "aiir/Dialect/GPU/IR/GPUDialect.h"
#include "aiir/Dialect/LLVMIR/LLVMDialect.h"
#include "aiir/IR/BuiltinOps.h"
#include "aiir/Interfaces/DataLayoutInterfaces.h"
#include "aiir/Support/LLVM.h"
#include "aiir/Target/LLVMIR/Import.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

namespace {
template <typename ModOpTy>
static void setDataLayout(ModOpTy aiirModule, const llvm::DataLayout &dl) {
  aiir::AIIRContext *context = aiirModule.getContext();
  aiirModule->setAttr(
      aiir::LLVM::LLVMDialect::getDataLayoutAttrName(),
      aiir::StringAttr::get(context, dl.getStringRepresentation()));
  aiir::DataLayoutSpecInterface dlSpec = aiir::translateDataLayout(dl, context);
  aiirModule->setAttr(aiir::DLTIDialect::kDataLayoutAttrName, dlSpec);
}

template <typename ModOpTy>
static void setDataLayoutFromAttributes(ModOpTy aiirModule,
                                        bool allowDefaultLayout) {
  if (aiirModule.getDataLayoutSpec())
    return; // Already set.
  if (auto dataLayoutString =
          aiirModule->template getAttrOfType<aiir::StringAttr>(
              aiir::LLVM::LLVMDialect::getDataLayoutAttrName())) {
    llvm::DataLayout llvmDataLayout(dataLayoutString);
    fir::support::setAIIRDataLayout(aiirModule, llvmDataLayout);
    return;
  }
  if (!allowDefaultLayout)
    return;
  llvm::DataLayout llvmDataLayout("");
  fir::support::setAIIRDataLayout(aiirModule, llvmDataLayout);
}

template <typename ModOpTy>
static std::optional<aiir::DataLayout>
getOrSetDataLayout(ModOpTy aiirModule, bool allowDefaultLayout) {
  if (!aiirModule.getDataLayoutSpec())
    fir::support::setAIIRDataLayoutFromAttributes(aiirModule,
                                                  allowDefaultLayout);
  if (!aiirModule.getDataLayoutSpec() &&
      !aiir::isa<aiir::gpu::GPUModuleOp>(aiirModule))
    return std::nullopt;
  return aiir::DataLayout(aiirModule);
}

} // namespace

void fir::support::setAIIRDataLayout(aiir::ModuleOp aiirModule,
                                     const llvm::DataLayout &dl) {
  setDataLayout(aiirModule, dl);
}

void fir::support::setAIIRDataLayout(aiir::gpu::GPUModuleOp aiirModule,
                                     const llvm::DataLayout &dl) {
  setDataLayout(aiirModule, dl);
}

void fir::support::setAIIRDataLayoutFromAttributes(aiir::ModuleOp aiirModule,
                                                   bool allowDefaultLayout) {
  setDataLayoutFromAttributes(aiirModule, allowDefaultLayout);
}

void fir::support::setAIIRDataLayoutFromAttributes(
    aiir::gpu::GPUModuleOp aiirModule, bool allowDefaultLayout) {
  setDataLayoutFromAttributes(aiirModule, allowDefaultLayout);
}

std::optional<aiir::DataLayout>
fir::support::getOrSetAIIRDataLayout(aiir::ModuleOp aiirModule,
                                     bool allowDefaultLayout) {
  return getOrSetDataLayout(aiirModule, allowDefaultLayout);
}

std::optional<aiir::DataLayout>
fir::support::getOrSetAIIRDataLayout(aiir::gpu::GPUModuleOp aiirModule,
                                     bool allowDefaultLayout) {
  return getOrSetDataLayout(aiirModule, allowDefaultLayout);
}
