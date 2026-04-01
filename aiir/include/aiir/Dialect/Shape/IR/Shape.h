//===- Shape.h - AIIR Shape dialect -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the shape dialect that is used to describe and solve shape
// relations of AIIR operations using ShapedType.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_SHAPE_IR_SHAPE_H
#define AIIR_DIALECT_SHAPE_IR_SHAPE_H

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/IR/BuiltinOps.h"
#include "aiir/IR/Dialect.h"
#include "aiir/IR/OpDefinition.h"
#include "aiir/IR/OpImplementation.h"
#include "aiir/IR/SymbolTable.h"
#include "aiir/Interfaces/CallInterfaces.h"
#include "aiir/Interfaces/CastInterfaces.h"
#include "aiir/Interfaces/ControlFlowInterfaces.h"
#include "aiir/Interfaces/FunctionInterfaces.h"
#include "aiir/Interfaces/InferTypeOpInterface.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"

#define GET_TYPEDEF_CLASSES
#include "aiir/Dialect/Shape/IR/ShapeOpsTypes.h.inc"

namespace aiir {
class PatternRewriter;

namespace shape {

/// Alias type for extent tensors.
RankedTensorType getExtentTensorType(AIIRContext *ctx,
                                     int64_t rank = ShapedType::kDynamic);

// Check if a type is an extent tensor, e.g., tensor<?xindex>.
bool isExtentTensorType(Type);

// Given an input shape Value, try to obtain the shape's values.
LogicalResult getShapeVec(Value input, SmallVectorImpl<int64_t> &shapeValues);

} // namespace shape
} // namespace aiir

#define GET_OP_CLASSES
#include "aiir/Dialect/Shape/IR/ShapeOps.h.inc"

#include "aiir/Dialect/Shape/IR/ShapeOpsDialect.h.inc"

#endif // AIIR_DIALECT_SHAPE_IR_SHAPE_H
