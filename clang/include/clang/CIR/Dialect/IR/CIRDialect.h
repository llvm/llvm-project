//===- CIRDialect.h - CIR dialect -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the CIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_DIALECT_IR_CIRDIALECT_H
#define LLVM_CLANG_CIR_DIALECT_IR_CIRDIALECT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIROpsDialect.h.inc"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIROpsStructs.h.inc"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

#include "clang/CIR/Interfaces/ASTAttrInterfaces.h"

namespace mlir {
namespace OpTrait {

namespace impl {
// These functions are out-of-line implementations of the methods in the
// corresponding trait classes.  This avoids them being template
// instantiated/duplicated.
LogicalResult verifySameFirstOperandAndResultType(Operation *op);
LogicalResult verifySameFirstSecondOperandAndResultType(Operation *op);
} // namespace impl

/// This class provides verification for ops that are known to have the same
/// first operand and result type.
///
template <typename ConcreteType>
class SameFirstOperandAndResultType
    : public TraitBase<ConcreteType, SameFirstOperandAndResultType> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifySameFirstOperandAndResultType(op);
  }
};

/// This class provides verification for ops that are known to have the same
/// first operand and result type.
///
template <typename ConcreteType>
class SameFirstSecondOperandAndResultType
    : public TraitBase<ConcreteType, SameFirstSecondOperandAndResultType> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifySameFirstSecondOperandAndResultType(op);
  }
};

} // namespace OpTrait

namespace cir {
void buildTerminatedBody(OpBuilder &builder, Location loc);
} // namespace cir

} // namespace mlir

#define GET_OP_CLASSES
#include "clang/CIR/Dialect/IR/CIROps.h.inc"

#endif // LLVM_CLANG_CIR_DIALECT_IR_CIRDIALECT_H

