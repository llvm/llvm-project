//===- MLProgram.h - MLProgram dialect ----------------------------*- C++-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef AIIR_DIALECT_MLPROGRAM_IR_MLPROGRAM_H_
#define AIIR_DIALECT_MLPROGRAM_IR_MLPROGRAM_H_

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/Dialect/MLProgram/IR/MLProgramAttributes.h"
#include "aiir/Dialect/MLProgram/IR/MLProgramTypes.h"
#include "aiir/IR/Dialect.h"
#include "aiir/IR/OpDefinition.h"
#include "aiir/IR/OpImplementation.h"
#include "aiir/IR/RegionKindInterface.h"
#include "aiir/IR/SymbolTable.h"
#include "aiir/Interfaces/CallInterfaces.h"
#include "aiir/Interfaces/ControlFlowInterfaces.h"
#include "aiir/Interfaces/FunctionInterfaces.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// MLProgramDialect
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/MLProgram/IR/MLProgramOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// MLProgram Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "aiir/Dialect/MLProgram/IR/MLProgramOps.h.inc"

#endif // AIIR_DIALECT_MLPROGRAM_IR_MLPROGRAM_H_
