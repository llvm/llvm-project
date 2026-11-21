//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains external OpenACC operation interfaces for CIR.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_OPENACC_CIROPENACCOPSINTERFACES_H
#define CLANG_CIR_DIALECT_OPENACC_CIROPENACCOPSINTERFACES_H

#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

namespace cir::acc {

struct AddressOfGlobalModel
    : public mlir::acc::AddressOfGlobalOpInterface::ExternalModel<
          AddressOfGlobalModel, cir::GetGlobalOp> {
  mlir::SymbolRefAttr getSymbol(mlir::Operation *op) const;
};

struct GlobalVariableModel
    : public mlir::acc::GlobalVariableOpInterface::ExternalModel<
          GlobalVariableModel, cir::GlobalOp> {
  bool isConstant(mlir::Operation *op) const;
  bool hasInitializer(mlir::Operation *op) const;
  mlir::Region *getInitRegion(mlir::Operation *op) const;
  bool isDeviceAccessible(mlir::Operation *op) const;
  bool isInDeviceMemory(mlir::Operation *op) const;
  bool isCompilerGenerated(mlir::Operation *op) const;
};

} // namespace cir::acc

#endif // CLANG_CIR_DIALECT_OPENACC_CIROPENACCOPSINTERFACES_H
