//===- FuncOps.h - Func Dialect Operations ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_FUNC_IR_OPS_H
#define AIIR_DIALECT_FUNC_IR_OPS_H

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/IR/Builders.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/Dialect.h"
#include "aiir/IR/OpImplementation.h"
#include "aiir/IR/SymbolTable.h"
#include "aiir/Interfaces/CallInterfaces.h"
#include "aiir/Interfaces/ControlFlowInterfaces.h"
#include "aiir/Interfaces/FunctionInterfaces.h"
#include "aiir/Interfaces/InferTypeOpInterface.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"

namespace aiir {
class PatternRewriter;
} // namespace aiir

#define GET_OP_CLASSES
#include "aiir/Dialect/Func/IR/FuncOps.h.inc"

#include "aiir/Dialect/Func/IR/FuncOpsDialect.h.inc"

namespace llvm {

/// Allow stealing the low bits of FuncOp.
template <>
struct PointerLikeTypeTraits<aiir::func::FuncOp> {
  static inline void *getAsVoidPointer(aiir::func::FuncOp val) {
    return const_cast<void *>(val.getAsOpaquePointer());
  }
  static inline aiir::func::FuncOp getFromVoidPointer(void *p) {
    return aiir::func::FuncOp::getFromOpaquePointer(p);
  }
  static constexpr int numLowBitsAvailable = 3;
};
} // namespace llvm

#endif // AIIR_DIALECT_FUNC_IR_OPS_H
