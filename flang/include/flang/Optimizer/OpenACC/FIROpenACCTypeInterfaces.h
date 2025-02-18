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
};

} // namespace fir::acc

#endif // FLANG_OPTIMIZER_OPENACC_FIROPENACCTYPEINTERFACES_H_
