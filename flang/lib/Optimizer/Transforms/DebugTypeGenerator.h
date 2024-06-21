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

  /// The 'genAllocated' is true when we want to generate 'allocated' field
  /// in the DICompositeType. It is needed for the allocatable arrays.
  /// Similarly, 'genAssociated' is used with 'pointer' type to generate
  /// 'associated' field.
  mlir::LLVM::DITypeAttr
  convertBoxedSequenceType(fir::SequenceType seqTy,
                           mlir::LLVM::DIFileAttr fileAttr,
                           mlir::LLVM::DIScopeAttr scope, mlir::Location loc,
                           bool genAllocated, bool genAssociated);
  mlir::LLVM::DITypeAttr convertCharacterType(fir::CharacterType charTy,
                                              mlir::LLVM::DIFileAttr fileAttr,
                                              mlir::LLVM::DIScopeAttr scope,
                                              mlir::Location loc);

  mlir::LLVM::DITypeAttr
  convertPointerLikeType(mlir::Type elTy, mlir::LLVM::DIFileAttr fileAttr,
                         mlir::LLVM::DIScopeAttr scope, mlir::Location loc,
                         bool genAllocated, bool genAssociated);

  mlir::ModuleOp module;
  KindMapping kindMapping;
  std::uint64_t dimsSize;
  std::uint64_t dimsOffset;
  std::uint64_t ptrSize;
};

} // namespace fir

#endif // FORTRAN_OPTIMIZER_TRANSFORMS_DEBUGTYPEGENERATOR_H
