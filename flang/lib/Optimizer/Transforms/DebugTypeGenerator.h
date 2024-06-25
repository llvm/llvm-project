//===-- DebugTypeGenerator.h -- type conversion ------------------- C++ -*-===//
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

#ifndef FORTRAN_OPTIMIZER_TRANSFORMS_DEBUGTYPEGENERATOR_H
#define FORTRAN_OPTIMIZER_TRANSFORMS_DEBUGTYPEGENERATOR_H

#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "llvm/Support/Debug.h"

namespace fir {

/// This converts FIR/mlir type to DITypeAttr.
class DebugTypeGenerator {
public:
  DebugTypeGenerator(mlir::ModuleOp module);

  mlir::LLVM::DITypeAttr convertType(mlir::Type Ty,
                                     mlir::LLVM::DIFileAttr fileAttr,
                                     mlir::LLVM::DIScopeAttr scope,
                                     mlir::Location loc);

private:
  mlir::LLVM::DITypeAttr convertSequenceType(fir::SequenceType seqTy,
                                             mlir::LLVM::DIFileAttr fileAttr,
                                             mlir::LLVM::DIScopeAttr scope,
                                             mlir::Location loc);
  mlir::ModuleOp module;
  KindMapping kindMapping;
};

} // namespace fir

#endif // FORTRAN_OPTIMIZER_TRANSFORMS_DEBUGTYPEGENERATOR_H
