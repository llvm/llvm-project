//===-- Optimizer/Support/DataLayout.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_SUPPORT_DATALAYOUT_H
#define FORTRAN_OPTIMIZER_SUPPORT_DATALAYOUT_H

namespace mlir {
class ModuleOp;
}
namespace llvm {
class DataLayout;
}

namespace fir::support {
/// Create an mlir::DataLayoutSpecInterface attribute from an llvm::DataLayout
/// and set it on the provided mlir::ModuleOp.
/// Also set the llvm.data_layout attribute with the string representation of
/// the llvm::DataLayout on the module.
/// These attributes are replaced if they were already set.
void setMLIRDataLayout(mlir::ModuleOp mlirModule, const llvm::DataLayout &dl);

/// Create an mlir::DataLayoutSpecInterface from the llvm.data_layout attribute
/// if one is provided. If such attribute is not available, create a default
/// target independent layout when allowDefaultLayout is true. Otherwise do
/// nothing.
void setMLIRDataLayoutFromAttributes(mlir::ModuleOp mlirModule,
                                     bool allowDefaultLayout);
} // namespace fir::support

#endif // FORTRAN_OPTIMIZER_SUPPORT_DATALAYOUT_H
