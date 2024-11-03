//===-- Optimizer/Dialect/FIROps.h - FIR operations -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_DIALECT_FIROPS_H
#define FORTRAN_OPTIMIZER_DIALECT_FIROPS_H

#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/FirAliasTagOpInterface.h"
#include "flang/Optimizer/Dialect/FortranVariableInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace fir {

class FirEndOp;
class DoLoopOp;
class RealAttr;

void buildCmpCOp(mlir::OpBuilder &builder, mlir::OperationState &result,
                 mlir::arith::CmpFPredicate predicate, mlir::Value lhs,
                 mlir::Value rhs);
unsigned getCaseArgumentOffset(llvm::ArrayRef<mlir::Attribute> cases,
                               unsigned dest);
DoLoopOp getForInductionVarOwner(mlir::Value val);
mlir::ParseResult isValidCaseAttr(mlir::Attribute attr);
mlir::ParseResult parseCmpcOp(mlir::OpAsmParser &parser,
                              mlir::OperationState &result);
mlir::ParseResult parseSelector(mlir::OpAsmParser &parser,
                                mlir::OperationState &result,
                                mlir::OpAsmParser::UnresolvedOperand &selector,
                                mlir::Type &type);

static constexpr llvm::StringRef getAdaptToByRefAttrName() {
  return "adapt.valuebyref";
}

static constexpr llvm::StringRef getNormalizedLowerBoundAttrName() {
  return "normalized.lb";
}

/// Model operations which affect global debugging information
struct DebuggingResource
    : public mlir::SideEffects::Resource::Base<DebuggingResource> {
  mlir::StringRef getName() final { return "DebuggingResource"; }
};

} // namespace fir

#define GET_OP_CLASSES
#include "flang/Optimizer/Dialect/FIROps.h.inc"

#endif // FORTRAN_OPTIMIZER_DIALECT_FIROPS_H
