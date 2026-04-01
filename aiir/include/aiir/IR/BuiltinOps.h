//===- BuiltinOps.h - AIIR Builtin Operations -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Builtin dialect's operations.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_IR_BUILTINOPS_H_
#define AIIR_IR_BUILTINOPS_H_

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/IR/OpImplementation.h"
#include "aiir/IR/OwningOpRef.h"
#include "aiir/IR/RegionKindInterface.h"
#include "aiir/IR/SymbolTable.h"
#include "aiir/Interfaces/DataLayoutInterfaces.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/PointerLikeTypeTraits.h"

//===----------------------------------------------------------------------===//
// Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "aiir/IR/BuiltinOps.h.inc"

namespace llvm {
/// Allow stealing the low bits of ModuleOp.
template <>
struct PointerLikeTypeTraits<aiir::ModuleOp> {
public:
  static inline void *getAsVoidPointer(aiir::ModuleOp val) {
    return const_cast<void *>(val.getAsOpaquePointer());
  }
  static inline aiir::ModuleOp getFromVoidPointer(void *p) {
    return aiir::ModuleOp::getFromOpaquePointer(p);
  }
  static constexpr int numLowBitsAvailable = 3;
};
} // namespace llvm

#endif // AIIR_IR_BUILTINOPS_H_
