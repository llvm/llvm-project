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

#include "flang/Optimizer/Dialect/FIROperationMoveOpInterface.h"
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
  mlir::Region *getInitRegion(mlir::Operation *op) const;
  bool isDeviceData(mlir::Operation *op) const;
};

template <typename Op>
struct IndirectGlobalAccessModel
    : public mlir::acc::IndirectGlobalAccessOpInterface::ExternalModel<
          IndirectGlobalAccessModel<Op>, Op> {
  void getReferencedSymbols(mlir::Operation *op,
                            llvm::SmallVectorImpl<mlir::SymbolRefAttr> &symbols,
                            mlir::SymbolTable *symbolTable) const;
};

/// External model for OutlineRematerializationOpInterface.
/// This interface marks operations that are candidates for rematerialization
/// during outlining. These operations produce synthetic types or values
/// that cannot be passed as arguments to outlined regions.
template <typename Op>
struct OutlineRematerializationModel
    : public mlir::acc::OutlineRematerializationOpInterface::ExternalModel<
          OutlineRematerializationModel<Op>, Op> {};

/// External model for OffloadRegionOpInterface.
/// This interface marks operations whose regions are targets for offloading
/// and outlining.
template <typename Op>
struct OffloadRegionModel
    : public mlir::acc::OffloadRegionOpInterface::ExternalModel<
          OffloadRegionModel<Op>, Op> {};

/// External model for fir::OperationMoveOpInterface.
/// This interface provides methods to identify whether
/// operations can be moved (e.g. by LICM, CSE, etc.) from/into
/// OpenACC dialect operations.
template <typename Op>
struct OperationMoveModel : public fir::OperationMoveOpInterface::ExternalModel<
                                OperationMoveModel<Op>, Op> {
  // Returns true if it is allowed to move the given 'candidate'
  // operation from the 'descendant' operation into 'op' operation.
  // If 'candidate' is nullptr, then the caller is querying whether
  // any operation from any descendant can be moved into 'op' operation.
  bool canMoveFromDescendant(mlir::Operation *op, mlir::Operation *descendant,
                             mlir::Operation *candidate) const;

  // Returns true if it is allowed to move the given 'candidate'
  // operation out of 'op' operation. If 'candidate' is nullptr,
  // then the caller is querying whether any operation can be moved
  // out of 'op' operation.
  bool canMoveOutOf(mlir::Operation *op, mlir::Operation *candidate) const;
};

} // namespace fir::acc

#endif // FLANG_OPTIMIZER_OPENACC_FIROPENACC_OPS_INTERFACES_H_
