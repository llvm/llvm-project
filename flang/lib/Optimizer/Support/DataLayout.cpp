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
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

namespace {
template <typename ModOpTy>
static void setDataLayout(ModOpTy mlirModule, const llvm::DataLayout &dl) {
  mlir::MLIRContext *context = mlirModule.getContext();
  mlirModule->setAttr(
      mlir::LLVM::LLVMDialect::getDataLayoutAttrName(),
      mlir::StringAttr::get(context, dl.getStringRepresentation()));
  mlir::DataLayoutSpecInterface dlSpec = mlir::translateDataLayout(dl, context);
  mlirModule->setAttr(mlir::DLTIDialect::kDataLayoutAttrName, dlSpec);
}

template <typename ModOpTy>
static void setDataLayoutFromAttributes(ModOpTy mlirModule,
                                        bool allowDefaultLayout) {
  if (mlirModule.getDataLayoutSpec())
    return; // Already set.
  if (auto dataLayoutString =
          mlirModule->template getAttrOfType<mlir::StringAttr>(
              mlir::LLVM::LLVMDialect::getDataLayoutAttrName())) {
    llvm::DataLayout llvmDataLayout(dataLayoutString);
    fir::support::setMLIRDataLayout(mlirModule, llvmDataLayout);
    return;
  }
  if (!allowDefaultLayout)
    return;
  llvm::DataLayout llvmDataLayout("");
  fir::support::setMLIRDataLayout(mlirModule, llvmDataLayout);
}

template <typename ModOpTy>
static std::optional<mlir::DataLayout>
getOrSetDataLayout(ModOpTy mlirModule, bool allowDefaultLayout) {
  if (!mlirModule.getDataLayoutSpec())
    fir::support::setMLIRDataLayoutFromAttributes(mlirModule,
                                                  allowDefaultLayout);
  if (!mlirModule.getDataLayoutSpec() &&
      !mlir::isa<mlir::gpu::GPUModuleOp>(mlirModule))
    return std::nullopt;
  return mlir::DataLayout(mlirModule);
}

} // namespace

void fir::support::setMLIRDataLayout(mlir::ModuleOp mlirModule,
                                     const llvm::DataLayout &dl) {
  setDataLayout(mlirModule, dl);
}

void fir::support::setMLIRDataLayout(mlir::gpu::GPUModuleOp mlirModule,
                                     const llvm::DataLayout &dl) {
  setDataLayout(mlirModule, dl);
}

void fir::support::setMLIRDataLayoutFromAttributes(mlir::ModuleOp mlirModule,
                                                   bool allowDefaultLayout) {
  setDataLayoutFromAttributes(mlirModule, allowDefaultLayout);
}

void fir::support::setMLIRDataLayoutFromAttributes(
    mlir::gpu::GPUModuleOp mlirModule, bool allowDefaultLayout) {
  setDataLayoutFromAttributes(mlirModule, allowDefaultLayout);
}

std::optional<mlir::DataLayout>
fir::support::getOrSetMLIRDataLayout(mlir::ModuleOp mlirModule,
                                     bool allowDefaultLayout) {
  return getOrSetDataLayout(mlirModule, allowDefaultLayout);
}

std::optional<mlir::DataLayout>
fir::support::getOrSetMLIRDataLayout(mlir::gpu::GPUModuleOp mlirModule,
                                     bool allowDefaultLayout) {
  return getOrSetDataLayout(mlirModule, allowDefaultLayout);
}
