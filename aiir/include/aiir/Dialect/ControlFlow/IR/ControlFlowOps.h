//===- ControlFlowOps.h - ControlFlow Operations ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the operations of the ControlFlow dialect.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_CONTROLFLOW_IR_CONTROLFLOWOPS_H
#define AIIR_DIALECT_CONTROLFLOW_IR_CONTROLFLOWOPS_H

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "aiir/IR/Builders.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/OpImplementation.h"
#include "aiir/Interfaces/ControlFlowInterfaces.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"

namespace aiir {
class PatternRewriter;
} // namespace aiir

#define GET_OP_CLASSES
#include "aiir/Dialect/ControlFlow/IR/ControlFlowOps.h.inc"

#endif // AIIR_DIALECT_CONTROLFLOW_IR_CONTROLFLOWOPS_H
