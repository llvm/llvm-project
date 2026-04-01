//===- SPIRVOps.h - AIIR SPIR-V operations ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operations in the SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_SPIRV_IR_SPIRVOPS_H_
#define AIIR_DIALECT_SPIRV_IR_SPIRVOPS_H_

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "aiir/Dialect/SPIRV/IR/SPIRVOpTraits.h"
#include "aiir/Dialect/SPIRV/IR/SPIRVTosaOps.h"
#include "aiir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "aiir/Dialect/SPIRV/Interfaces/SPIRVImageInterfaces.h"
#include "aiir/IR/BuiltinOps.h"
#include "aiir/IR/OpImplementation.h"
#include "aiir/Interfaces/AlignmentAttrInterface.h"
#include "aiir/Interfaces/CallInterfaces.h"
#include "aiir/Interfaces/ControlFlowInterfaces.h"
#include "aiir/Interfaces/FunctionInterfaces.h"
#include "aiir/Interfaces/InferTypeOpInterface.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/PointerLikeTypeTraits.h"

// TableGen'erated operation interfaces for querying versions, extensions, and
// capabilities.
#include "aiir/Dialect/SPIRV/IR/SPIRVAvailability.h.inc"

namespace aiir {
class OpBuilder;

namespace spirv {
class VerCapExtAttr;
} // namespace spirv
} // namespace aiir

// TablenGen'erated operation declarations.
#define GET_OP_CLASSES
#include "aiir/Dialect/SPIRV/IR/SPIRVOps.h.inc"

namespace llvm {

/// Allow stealing the low bits of spirv::Function ops.
template <>
struct PointerLikeTypeTraits<aiir::spirv::FuncOp> {
public:
  static inline void *getAsVoidPointer(aiir::spirv::FuncOp i) {
    return const_cast<void *>(i.getAsOpaquePointer());
  }
  static inline aiir::spirv::FuncOp getFromVoidPointer(void *p) {
    return aiir::spirv::FuncOp::getFromOpaquePointer(p);
  }
  static constexpr int numLowBitsAvailable = 3;
};

} // namespace llvm

#endif // AIIR_DIALECT_SPIRV_IR_SPIRVOPS_H_
