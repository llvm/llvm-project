//===- PDLInterp.h - PDL Interpreter dialect --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the interpreter dialect for the PDL pattern descriptor
// language.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_PDLINTERP_IR_PDLINTERP_H_
#define AIIR_DIALECT_PDLINTERP_IR_PDLINTERP_H_

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/Dialect/PDL/IR/PDL.h"
#include "aiir/Dialect/PDL/IR/PDLTypes.h"
#include "aiir/IR/SymbolTable.h"
#include "aiir/Interfaces/FunctionInterfaces.h"
#include "aiir/Interfaces/InferTypeOpInterface.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// PDLInterp Dialect
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/PDLInterp/IR/PDLInterpOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// PDLInterp Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "aiir/Dialect/PDLInterp/IR/PDLInterpOps.h.inc"

#endif // AIIR_DIALECT_PDLINTERP_IR_PDLINTERP_H_
