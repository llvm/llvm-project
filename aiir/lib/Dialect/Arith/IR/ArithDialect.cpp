//===- ArithDialect.cpp - AIIR Arith dialect implementation -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Conversion/ConvertToEmitC/ToEmitCInterface.h"
#include "aiir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "aiir/Dialect/Arith/IR/Arith.h"
#include "aiir/Dialect/Bufferization/IR/BufferDeallocationOpInterface.h"
#include "aiir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "aiir/Dialect/UB/IR/UBOps.h"
#include "aiir/IR/Builders.h"
#include "aiir/IR/DialectImplementation.h"
#include "aiir/Interfaces/ValueBoundsOpInterface.h"
#include "aiir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace aiir;
using namespace aiir::arith;

#include "aiir/Dialect/Arith/IR/ArithOpsDialect.cpp.inc"
#include "aiir/Dialect/Arith/IR/ArithOpsInterfaces.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "aiir/Dialect/Arith/IR/ArithOpsAttributes.cpp.inc"

namespace {
/// This class defines the interface for handling inlining for arithmetic
/// dialect operations.
struct ArithInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// All arithmetic dialect ops can be inlined.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
};
} // namespace

void arith::ArithDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "aiir/Dialect/Arith/IR/ArithOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "aiir/Dialect/Arith/IR/ArithOpsAttributes.cpp.inc"
      >();
  addInterfaces<ArithInlinerInterface>();
  declarePromisedInterface<ConvertToEmitCPatternInterface, ArithDialect>();
  declarePromisedInterface<ConvertToLLVMPatternInterface, ArithDialect>();
  declarePromisedInterface<bufferization::BufferDeallocationOpInterface,
                           SelectOp>();
  declarePromisedInterfaces<bufferization::BufferizableOpInterface, ConstantOp,
                            IndexCastOp, SelectOp>();
  declarePromisedInterfaces<ValueBoundsOpInterface, AddIOp, ConstantOp, SubIOp,
                            MulIOp>();
}

/// Materialize an integer or floating point constant.
Operation *arith::ArithDialect::materializeConstant(OpBuilder &builder,
                                                    Attribute value, Type type,
                                                    Location loc) {
  if (auto poison = dyn_cast<ub::PoisonAttr>(value))
    return ub::PoisonOp::create(builder, loc, type, poison);

  return ConstantOp::materialize(builder, value, type, loc);
}
