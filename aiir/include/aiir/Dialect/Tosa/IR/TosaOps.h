//===-- TosaOps.h - TOSA dialect operation definitions ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the TOSA Dialect in AIIR.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_TOSA_IR_TOSAOPS_H
#define AIIR_DIALECT_TOSA_IR_TOSAOPS_H

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/Dialect/Quant/IR/QuantTypes.h"
#include "aiir/Dialect/Traits.h"
#include "aiir/IR/Matchers.h"
#include "aiir/IR/OpDefinition.h"
#include "aiir/IR/OpImplementation.h"
#include "aiir/IR/TypeUtilities.h"
#include "aiir/Interfaces/InferTypeOpInterface.h"
#include "aiir/Interfaces/LoopLikeInterface.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"
#include "aiir/Interfaces/VectorInterfaces.h"

//===----------------------------------------------------------------------===//
// TOSA dialect and structs includes.
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Tosa/IR/TosaEnums.h.inc"
#include "aiir/Dialect/Tosa/IR/TosaOpsDialect.h.inc"
#include "aiir/Transforms/DialectConversion.h"

//===----------------------------------------------------------------------===//
// TOSA operation validation includes.
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Tosa/IR/TosaAvailability.h.inc"

namespace aiir {
class PatternRewriter;

namespace tosa {

ParseResult parseVariableOpTypeOrInitialValue(OpAsmParser &parser,
                                              DenseElementsAttr &varShapeAttr,
                                              TypeAttr &typeAttr,
                                              Attribute &initialValueAttr);
void printVariableOpTypeOrInitialValue(OpAsmPrinter &p, Operation *op,
                                       DenseElementsAttr varShapeAttr,
                                       TypeAttr typeAttr,
                                       Attribute initialValueAttr);

#include "aiir/Dialect/Tosa/IR/TosaInterfaces.h.inc"

} // namespace tosa

namespace OpTrait {
namespace tosa {

// This trait verifies if the element type amoung operands and result
// of multiplication match tosa specification.
template <typename ConcreteType>
class MulOperandsAndResultElementType
    : public TraitBase<ConcreteType, MulOperandsAndResultElementType> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    // Check we have a single result.
    if (failed(impl::verifyOneResult(op)))
      return failure();
    Type resElemType = getElementTypeOrSelf(op->getResult(0));

    // Check we have lhs and rhs.
    if (failed(impl::verifyAtLeastNOperands(op, 2)))
      return failure();

    Type lhsElemType = getElementTypeOrSelf(op->getOperand(0));
    Type rhsElemType = getElementTypeOrSelf(op->getOperand(1));

    // Check that for i32 a shift has been explicitly provided.
    if (lhsElemType.isInteger(32) && failed(impl::verifyNOperands(op, 3)))
      return failure();

    // Verify operands type match (ignoring the shift parameter which will
    // always be i8).
    if (lhsElemType != rhsElemType)
      return op->emitOpError("requires the same element type for all operands");

    // Though the spec requires the element type of result to be i32, a more
    // relaxed way is provided at dialect level for easier cooperating with
    // other dialects.
    if (auto resIntType = dyn_cast<IntegerType>(resElemType)) {
      auto lhsIntType = cast<IntegerType>(lhsElemType);
      if (lhsIntType.getWidth() > resIntType.getWidth())
        return op->emitOpError("invalid data type size for operands or result");
    } else {
      // In cases of floating point type or quant types, op requires the same
      // element type for all operands and result (excluding shift).
      if (resElemType != lhsElemType)
        return op->emitOpError(
            "requires the same element type for all operands and results");
    }

    return llvm::success();
  }
};

/// This class indicates that an op is tosa-elementwise (permits broadcasting,
/// unlike Elementwise trait).
template <typename ConcreteType>
class TosaElementwiseOperator
    : public TraitBase<ConcreteType, TosaElementwiseOperator> {};

LogicalResult verifyTosaResolvableShapeOperands(Operation *op);
/// This class verifies that tosa shape operands are compile time resolvable
template <typename ConcreteType>
class TosaResolvableShapeOperands
    : public TraitBase<ConcreteType, TosaResolvableShapeOperands> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return verifyTosaResolvableShapeOperands(op);
  }
};

/// This class indicates that op operates on tosa shape types
template <typename ConcreteType>
class TosaShapeOperator : public TraitBase<ConcreteType, TosaShapeOperator> {};

LogicalResult verifyTosaShapeOperatorWithSameRanks(Operation *op);
/// This class indicates that op operates on tosa shape types
template <typename ConcreteType>
class TosaShapeOperatorWithSameRanks
    : public TraitBase<ConcreteType, TosaShapeOperatorWithSameRanks> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return verifyTosaShapeOperatorWithSameRanks(op);
  }
};

} // namespace tosa
} // namespace OpTrait

namespace tosa {

bool isa_tosa_shape_type(aiir::Type t);

/// Represents a dimension in the shape of a tensor that can be inferred
/// based on the other provided dimensions. For example, in a reshape
/// operation, -1 can be used to indicate a size that is the remainder
/// of the other dimensions.
constexpr int64_t kInferableDimSize = -1;

} // namespace tosa

} // namespace aiir

#define GET_ATTRDEF_CLASSES
#include "aiir/Dialect/Tosa/IR/TosaAttributes.h.inc"

#define GET_TYPEDEF_CLASSES
#include "aiir/Dialect/Tosa/IR/TosaOpsTypesBase.h.inc"

#define GET_OP_CLASSES
#include "aiir/Dialect/Tosa/IR/TosaOps.h.inc"

namespace aiir {
namespace tosa {

// Create a rank-1 const tensor for zero point of the source tensor.
std::optional<Value> createZeroPointTensor(OpBuilder &builder, Location loc,
                                           Type srcElemType, int64_t zp = 0);

// Create a pad-const const tensor with value of `val` of required data-type
Value createPadConstTensor(OpBuilder &builder, Location loc, Value src,
                           int32_t val = 0);

// returns type of variable op
RankedTensorType getVariableType(VariableOp variableOp);

// Returns the bitwidth of a TOSA tensor element type
unsigned getBitWidth(Type type);

} // namespace tosa
} // namespace aiir

#endif // AIIR_DIALECT_TOSA_IR_TOSAOPS_H
