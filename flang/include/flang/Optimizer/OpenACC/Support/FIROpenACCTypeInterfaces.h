//===- FIROpenACCTypeInterfaces.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains external dialect interfaces for FIR.
//
//===----------------------------------------------------------------------===//

#ifndef FLANG_OPTIMIZER_OPENACC_FIROPENACCTYPEINTERFACES_H_
#define FLANG_OPTIMIZER_OPENACC_FIROPENACCTYPEINTERFACES_H_

#include "flang/Optimizer/Dialect/FIRType.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"

namespace fir::acc {

template <typename T>
struct OpenACCPointerLikeModel
    : public mlir::acc::PointerLikeType::ExternalModel<
          OpenACCPointerLikeModel<T>, T> {
  mlir::Type getElementType(mlir::Type pointer) const {
    return mlir::cast<T>(pointer).getElementType();
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
};

template <typename T>
struct OpenACCMappableModel
    : public mlir::acc::MappableType::ExternalModel<OpenACCMappableModel<T>,
                                                    T> {
  mlir::TypedValue<mlir::acc::PointerLikeType> getVarPtr(::mlir::Type type,
                                                         mlir::Value var) const;

  std::optional<llvm::TypeSize>
  getSizeInBytes(mlir::Type type, mlir::Value var, mlir::ValueRange accBounds,
                 const mlir::DataLayout &dataLayout) const;

  std::optional<int64_t>
  getOffsetInBytes(mlir::Type type, mlir::Value var, mlir::ValueRange accBounds,
                   const mlir::DataLayout &dataLayout) const;

  llvm::SmallVector<mlir::Value>
  generateAccBounds(mlir::Type type, mlir::Value var,
                    mlir::OpBuilder &builder) const;

  mlir::acc::VariableTypeCategory getTypeCategory(mlir::Type type,
                                                  mlir::Value var) const;

  mlir::Value generatePrivateInit(mlir::Type type, mlir::OpBuilder &builder,
                                  mlir::Location loc,
                                  mlir::TypedValue<mlir::acc::MappableType> var,
                                  llvm::StringRef varName,
                                  mlir::ValueRange extents, mlir::Value initVal,
                                  bool &needsDestroy) const;

  bool generatePrivateDestroy(mlir::Type type, mlir::OpBuilder &builder,
                              mlir::Location loc, mlir::Value privatized) const;
};

} // namespace fir::acc

#endif // FLANG_OPTIMIZER_OPENACC_FIROPENACCTYPEINTERFACES_H_
