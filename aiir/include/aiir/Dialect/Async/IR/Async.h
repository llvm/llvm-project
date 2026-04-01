//===- Async.h - AIIR Async dialect -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the async dialect that is used for modeling asynchronous
// execution.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_ASYNC_IR_ASYNC_H
#define AIIR_DIALECT_ASYNC_IR_ASYNC_H

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/Dialect/Async/IR/AsyncTypes.h"
#include "aiir/IR/Builders.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/Dialect.h"
#include "aiir/IR/OpImplementation.h"
#include "aiir/IR/PatternMatch.h"
#include "aiir/IR/SymbolTable.h"
#include "aiir/Interfaces/CallInterfaces.h"
#include "aiir/Interfaces/ControlFlowInterfaces.h"
#include "aiir/Interfaces/FunctionInterfaces.h"
#include "aiir/Interfaces/InferTypeOpInterface.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// Async Dialect
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Async/IR/AsyncOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Async Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "aiir/Dialect/Async/IR/AsyncOps.h.inc"

//===----------------------------------------------------------------------===//
// Helper functions of Async dialect transformations.
//===----------------------------------------------------------------------===//

namespace aiir {
namespace async {

/// Returns true if the type is reference counted at runtime.
inline bool isRefCounted(Type type) {
  return isa<TokenType, ValueType, GroupType>(type);
}

} // namespace async
} // namespace aiir

namespace llvm {

/// Allow stealing the low bits of async::FuncOp.
template <>
struct PointerLikeTypeTraits<aiir::async::FuncOp> {
  static inline void *getAsVoidPointer(aiir::async::FuncOp val) {
    return const_cast<void *>(val.getAsOpaquePointer());
  }
  static inline aiir::async::FuncOp getFromVoidPointer(void *p) {
    return aiir::async::FuncOp::getFromOpaquePointer(p);
  }
  static constexpr int numLowBitsAvailable = 3;
};
} // namespace llvm

#endif // AIIR_DIALECT_ASYNC_IR_ASYNC_H
