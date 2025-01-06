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

#include "flang/Optimizer/CodeGen/CGOps.h"
#include "flang/Optimizer/CodeGen/TypeConverter.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/Support/DataLayout.h"
#include "llvm/Support/Debug.h"

namespace fir {

/// This converts FIR/mlir type to DITypeAttr.
class DebugTypeGenerator {
public:
  DebugTypeGenerator(mlir::ModuleOp module, mlir::SymbolTable *symbolTable,
                     const mlir::DataLayout &dl);

  mlir::LLVM::DITypeAttr convertType(mlir::Type Ty,
                                     mlir::LLVM::DIFileAttr fileAttr,
                                     mlir::LLVM::DIScopeAttr scope,
                                     fir::cg::XDeclareOp declOp);

private:
  mlir::LLVM::DITypeAttr convertRecordType(fir::RecordType Ty,
                                           mlir::LLVM::DIFileAttr fileAttr,
                                           mlir::LLVM::DIScopeAttr scope,
                                           fir::cg::XDeclareOp declOp);
  mlir::LLVM::DITypeAttr convertTupleType(mlir::TupleType Ty,
                                          mlir::LLVM::DIFileAttr fileAttr,
                                          mlir::LLVM::DIScopeAttr scope,
                                          fir::cg::XDeclareOp declOp);
  mlir::LLVM::DITypeAttr convertSequenceType(fir::SequenceType seqTy,
                                             mlir::LLVM::DIFileAttr fileAttr,
                                             mlir::LLVM::DIScopeAttr scope,
                                             fir::cg::XDeclareOp declOp);
  mlir::LLVM::DITypeAttr convertVectorType(fir::VectorType vecTy,
                                           mlir::LLVM::DIFileAttr fileAttr,
                                           mlir::LLVM::DIScopeAttr scope,
                                           fir::cg::XDeclareOp declOp);

  /// The 'genAllocated' is true when we want to generate 'allocated' field
  /// in the DICompositeType. It is needed for the allocatable arrays.
  /// Similarly, 'genAssociated' is used with 'pointer' type to generate
  /// 'associated' field.
  mlir::LLVM::DITypeAttr convertBoxedSequenceType(
      fir::SequenceType seqTy, mlir::LLVM::DIFileAttr fileAttr,
      mlir::LLVM::DIScopeAttr scope, fir::cg::XDeclareOp declOp,
      bool genAllocated, bool genAssociated);
  mlir::LLVM::DITypeAttr convertCharacterType(fir::CharacterType charTy,
                                              mlir::LLVM::DIFileAttr fileAttr,
                                              mlir::LLVM::DIScopeAttr scope,
                                              fir::cg::XDeclareOp declOp,
                                              bool hasDescriptor);

  mlir::LLVM::DITypeAttr convertPointerLikeType(mlir::Type elTy,
                                                mlir::LLVM::DIFileAttr fileAttr,
                                                mlir::LLVM::DIScopeAttr scope,
                                                fir::cg::XDeclareOp declOp,
                                                bool genAllocated,
                                                bool genAssociated);
  mlir::LLVM::DILocalVariableAttr
  generateArtificialVariable(mlir::MLIRContext *context, mlir::Value Val,
                             mlir::LLVM::DIFileAttr fileAttr,
                             mlir::LLVM::DIScopeAttr scope,
                             fir::cg::XDeclareOp declOp);
  std::pair<std::uint64_t, unsigned short>
  getFieldSizeAndAlign(mlir::Type fieldTy);

  mlir::ModuleOp module;
  mlir::SymbolTable *symbolTable;
  const mlir::DataLayout *dataLayout;
  KindMapping kindMapping;
  fir::LLVMTypeConverter llvmTypeConverter;
  std::uint64_t dimsSize;
  std::uint64_t dimsOffset;
  std::uint64_t ptrSize;
  std::uint64_t lenOffset;
  std::uint64_t rankOffset;
  std::uint64_t rankSize;
  llvm::DenseMap<mlir::Type, mlir::LLVM::DITypeAttr> typeCache;
};

} // namespace fir

static uint32_t getLineFromLoc(mlir::Location loc) {
  uint32_t line = 1;
  if (auto fileLoc = mlir::dyn_cast<mlir::FileLineColLoc>(loc))
    line = fileLoc.getLine();
  return line;
}

#endif // FORTRAN_OPTIMIZER_TRANSFORMS_DEBUGTYPEGENERATOR_H
