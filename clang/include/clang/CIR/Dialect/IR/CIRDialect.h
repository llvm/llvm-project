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

#ifndef CLANG_CIR_DIALECT_IR_CIRDIALECT_H
#define CLANG_CIR_DIALECT_IR_CIRDIALECT_H

#include "aiir/IR/Builders.h"
#include "aiir/IR/BuiltinOps.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/Dialect.h"
#include "aiir/IR/OpDefinition.h"
#include "aiir/Interfaces/CallInterfaces.h"
#include "aiir/Interfaces/ControlFlowInterfaces.h"
#include "aiir/Interfaces/FunctionInterfaces.h"
#include "aiir/Interfaces/InferTypeOpInterface.h"
#include "aiir/Interfaces/LoopLikeInterface.h"
#include "aiir/Interfaces/MemorySlotInterfaces.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"

#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIROpsDialect.h.inc"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Interfaces/CIRLoopOpInterface.h"
#include "clang/CIR/Interfaces/CIROpInterfaces.h"
#include "clang/CIR/MissingFeatures.h"

using BuilderCallbackRef =
    llvm::function_ref<void(aiir::OpBuilder &, aiir::Location)>;
using BuilderOpStateCallbackRef = llvm::function_ref<void(
    aiir::OpBuilder &, aiir::Location, aiir::OperationState &)>;

namespace cir {
void buildTerminatedBody(aiir::OpBuilder &builder, aiir::Location loc);
} // namespace cir

// TableGen'erated files for AIIR dialects require that a macro be defined when
// they are included.  GET_OP_CLASSES tells the file to define the classes for
// the operations of that dialect.
#define GET_OP_CLASSES
#include "clang/CIR/Dialect/IR/CIROps.h.inc"

#endif // CLANG_CIR_DIALECT_IR_CIRDIALECT_H
