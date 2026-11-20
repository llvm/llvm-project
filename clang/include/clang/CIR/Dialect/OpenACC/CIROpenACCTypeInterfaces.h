//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains external OpenACC type interfaces for CIR.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_OPENACC_CIROPENACCTYPEINTERFACES_H
#define CLANG_CIR_DIALECT_OPENACC_CIROPENACCTYPEINTERFACES_H

#include "mlir/Dialect/OpenACC/OpenACC.h"

namespace cir::acc {

template <typename T>
struct OpenACCPointerLikeModel
    : public mlir::acc::PointerLikeType::ExternalModel<
          OpenACCPointerLikeModel<T>, T> {
  mlir::Type getElementType(mlir::Type pointer) const {
    return mlir::cast<T>(pointer).getPointee();
  }
  mlir::acc::VariableTypeCategory
  getPointeeTypeCategory(mlir::Type pointer,
                         mlir::TypedValue<mlir::acc::PointerLikeType> varPtr,
                         mlir::Type varType) const;

  mlir::Value genAllocate(mlir::Type pointer, mlir::OpBuilder &builder,
                          mlir::Location loc, llvm::StringRef varName,
                          mlir::Type varType, mlir::Value originalVar,
                          bool &needsFree) const;

  bool genFree(mlir::Type pointer, mlir::OpBuilder &builder, mlir::Location loc,
               mlir::TypedValue<mlir::acc::PointerLikeType> varToFree,
               mlir::Value allocRes, mlir::Type varType) const;

  bool genCopy(mlir::Type pointer, mlir::OpBuilder &builder, mlir::Location loc,
               mlir::TypedValue<mlir::acc::PointerLikeType> destination,
               mlir::TypedValue<mlir::acc::PointerLikeType> source,
               mlir::Type varType) const;

  mlir::Value genLoad(mlir::Type pointer, mlir::OpBuilder &builder,
                      mlir::Location loc,
                      mlir::TypedValue<mlir::acc::PointerLikeType> srcPtr,
                      mlir::Type valueType) const;

  bool genStore(mlir::Type pointer, mlir::OpBuilder &builder,
                mlir::Location loc, mlir::Value valueToStore,
                mlir::TypedValue<mlir::acc::PointerLikeType> destPtr) const;

  mlir::Value genCast(mlir::Type pointer, mlir::OpBuilder &builder,
                      mlir::Location loc, mlir::Value value,
                      mlir::Type resultType) const;

  mlir::MemRefType getAsMemRefType(mlir::Type pointer,
                                   mlir::ModuleOp module) const;

  bool isDeviceAccessible(mlir::Type pointer, mlir::Value var,
                          mlir::SymbolTableCollection *symbolTables) const;

  bool isInDeviceMemory(mlir::Type pointer, mlir::Value var,
                        mlir::SymbolTableCollection *symbolTables) const;
};

} // namespace cir::acc

#endif // CLANG_CIR_DIALECT_OPENACC_CIROPENACCTYPEINTERFACES_H
