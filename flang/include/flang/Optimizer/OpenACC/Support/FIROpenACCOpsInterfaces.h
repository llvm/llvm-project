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
#include "flang/Optimizer/Dialect/FortranVariableInterface.h"
#include "aiir/Dialect/OpenACC/OpenACC.h"

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
    : public aiir::acc::PartialEntityAccessOpInterface::ExternalModel<
          PartialEntityAccessModel<Op>, Op> {
  aiir::Value getBaseEntity(aiir::Operation *op) const;

  // Default implementation - returns false (partial view)
  bool isCompleteView(aiir::Operation *op) const { return false; }
};

// Full specializations for declare operations
template <>
struct PartialEntityAccessModel<fir::DeclareOp>
    : public aiir::acc::PartialEntityAccessOpInterface::ExternalModel<
          PartialEntityAccessModel<fir::DeclareOp>, fir::DeclareOp> {
  aiir::Value getBaseEntity(aiir::Operation *op) const;
  bool isCompleteView(aiir::Operation *op) const;
};

template <>
struct PartialEntityAccessModel<hlfir::DeclareOp>
    : public aiir::acc::PartialEntityAccessOpInterface::ExternalModel<
          PartialEntityAccessModel<hlfir::DeclareOp>, hlfir::DeclareOp> {
  aiir::Value getBaseEntity(aiir::Operation *op) const;
  bool isCompleteView(aiir::Operation *op) const;
};

struct AddressOfGlobalModel
    : public aiir::acc::AddressOfGlobalOpInterface::ExternalModel<
          AddressOfGlobalModel, fir::AddrOfOp> {
  aiir::SymbolRefAttr getSymbol(aiir::Operation *op) const;
};

struct GlobalVariableModel
    : public aiir::acc::GlobalVariableOpInterface::ExternalModel<
          GlobalVariableModel, fir::GlobalOp> {
  bool isConstant(aiir::Operation *op) const;
  aiir::Region *getInitRegion(aiir::Operation *op) const;
  bool isDeviceData(aiir::Operation *op) const;
};

template <typename Op>
struct IndirectGlobalAccessModel
    : public aiir::acc::IndirectGlobalAccessOpInterface::ExternalModel<
          IndirectGlobalAccessModel<Op>, Op> {
  void getReferencedSymbols(aiir::Operation *op,
                            llvm::SmallVectorImpl<aiir::SymbolRefAttr> &symbols,
                            aiir::SymbolTable *symbolTable) const;
};

/// External model for OutlineRematerializationOpInterface.
/// This interface marks operations that are candidates for rematerialization
/// during outlining. These operations produce synthetic types or values
/// that cannot be passed as arguments to outlined regions.
template <typename Op>
struct OutlineRematerializationModel
    : public aiir::acc::OutlineRematerializationOpInterface::ExternalModel<
          OutlineRematerializationModel<Op>, Op> {};

/// External model for OffloadRegionOpInterface.
/// This interface marks operations whose regions are targets for offloading
/// and outlining.
template <typename Op>
struct OffloadRegionModel
    : public aiir::acc::OffloadRegionOpInterface::ExternalModel<
          OffloadRegionModel<Op>, Op> {
  aiir::Region &getOffloadRegion(aiir::Operation *op) const {
    return aiir::cast<Op>(op).getRegion();
  }
};

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
  bool canMoveFromDescendant(aiir::Operation *op, aiir::Operation *descendant,
                             aiir::Operation *candidate) const;

  // Returns true if it is allowed to move the given 'candidate'
  // operation out of 'op' operation. If 'candidate' is nullptr,
  // then the caller is querying whether any operation can be moved
  // out of 'op' operation.
  bool canMoveOutOf(aiir::Operation *op, aiir::Operation *candidate) const;
};

struct ReductionInitOpFortranObjectViewModel
    : public fir::FortranObjectViewOpInterface::ExternalModel<
          ReductionInitOpFortranObjectViewModel, aiir::acc::ReductionInitOp> {
  aiir::Value getViewSource(aiir::Operation *op,
                            aiir::OpResult resultView) const;
  std::optional<std::int64_t> getViewOffset(aiir::Operation *op,
                                            aiir::OpResult resultView) const;
};

} // namespace fir::acc

#endif // FLANG_OPTIMIZER_OPENACC_FIROPENACC_OPS_INTERFACES_H_
