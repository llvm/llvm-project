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
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIROpsDialect.h.inc"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Interfaces/CIRLoopOpInterface.h"
#include "clang/CIR/Interfaces/CIROpInterfaces.h"
#include "clang/CIR/MissingFeatures.h"

using BuilderCallbackRef =
    llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)>;
using BuilderOpStateCallbackRef = llvm::function_ref<void(
    mlir::OpBuilder &, mlir::Location, mlir::OperationState &)>;

namespace cir {
void buildTerminatedBody(mlir::OpBuilder &builder, mlir::Location loc);
} // namespace cir

// TableGen'erated files for MLIR dialects require that a macro be defined when
// they are included.  GET_OP_CLASSES tells the file to define the classes for
// the operations of that dialect.
#define GET_OP_CLASSES
#include "clang/CIR/Dialect/IR/CIROps.h.inc"

#endif // LLVM_CLANG_CIR_DIALECT_IR_CIRDIALECT_H
