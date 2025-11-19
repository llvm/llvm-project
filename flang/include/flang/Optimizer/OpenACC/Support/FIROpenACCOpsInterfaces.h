//===- FIROpenACCOpsInterfaces.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains external operation interfaces for FIR.
//
//===----------------------------------------------------------------------===//

#ifndef FLANG_OPTIMIZER_OPENACC_FIROPENACC_OPS_INTERFACES_H_
#define FLANG_OPTIMIZER_OPENACC_FIROPENACC_OPS_INTERFACES_H_

#include "mlir/Dialect/OpenACC/OpenACC.h"

namespace fir {
class AddrOfOp;
class DeclareOp;
class GlobalOp;
} // namespace fir

namespace hlfir {
class DeclareOp;
class DesignateOp;
} // namespace hlfir

namespace fir::acc {

template <typename Op>
struct PartialEntityAccessModel
    : public mlir::acc::PartialEntityAccessOpInterface::ExternalModel<
          PartialEntityAccessModel<Op>, Op> {
  mlir::Value getBaseEntity(mlir::Operation *op) const;

  // Default implementation - returns false (partial view)
  bool isCompleteView(mlir::Operation *op) const { return false; }
};

// Full specializations for declare operations
template <>
struct PartialEntityAccessModel<fir::DeclareOp>
    : public mlir::acc::PartialEntityAccessOpInterface::ExternalModel<
          PartialEntityAccessModel<fir::DeclareOp>, fir::DeclareOp> {
  mlir::Value getBaseEntity(mlir::Operation *op) const;
  bool isCompleteView(mlir::Operation *op) const;
};

template <>
struct PartialEntityAccessModel<hlfir::DeclareOp>
    : public mlir::acc::PartialEntityAccessOpInterface::ExternalModel<
          PartialEntityAccessModel<hlfir::DeclareOp>, hlfir::DeclareOp> {
  mlir::Value getBaseEntity(mlir::Operation *op) const;
  bool isCompleteView(mlir::Operation *op) const;
};

struct AddressOfGlobalModel
    : public mlir::acc::AddressOfGlobalOpInterface::ExternalModel<
          AddressOfGlobalModel, fir::AddrOfOp> {
  mlir::SymbolRefAttr getSymbol(mlir::Operation *op) const;
};

struct GlobalVariableModel
    : public mlir::acc::GlobalVariableOpInterface::ExternalModel<
          GlobalVariableModel, fir::GlobalOp> {
  bool isConstant(mlir::Operation *op) const;
};

} // namespace fir::acc

#endif // FLANG_OPTIMIZER_OPENACC_FIROPENACC_OPS_INTERFACES_H_
