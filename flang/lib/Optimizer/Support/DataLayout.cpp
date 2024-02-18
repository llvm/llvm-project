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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

void fir::support::setMLIRDataLayout(mlir::ModuleOp mlirModule,
                                     const llvm::DataLayout &dl) {
  mlir::MLIRContext *context = mlirModule.getContext();
  mlirModule->setAttr(
      mlir::LLVM::LLVMDialect::getDataLayoutAttrName(),
      mlir::StringAttr::get(context, dl.getStringRepresentation()));
  mlir::DataLayoutSpecInterface dlSpec = mlir::translateDataLayout(dl, context);
  mlirModule->setAttr(mlir::DLTIDialect::kDataLayoutAttrName, dlSpec);
}

void fir::support::setMLIRDataLayoutFromAttributes(mlir::ModuleOp mlirModule,
                                                   bool allowDefaultLayout) {
  if (mlirModule.getDataLayoutSpec())
    return; // Already set.
  if (auto dataLayoutString = mlirModule->getAttrOfType<mlir::StringAttr>(
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

std::optional<mlir::DataLayout>
fir::support::getOrSetDataLayout(mlir::ModuleOp mlirModule,
                                 bool allowDefaultLayout) {
  if (!mlirModule.getDataLayoutSpec()) {
    fir::support::setMLIRDataLayoutFromAttributes(mlirModule,
                                                  allowDefaultLayout);
    if (!mlirModule.getDataLayoutSpec()) {
      return std::nullopt;
    }
  }
  return mlir::DataLayout(mlirModule);
}
