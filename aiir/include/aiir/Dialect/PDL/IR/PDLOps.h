//===- PDLOps.h - Pattern Descriptor Language Operations --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the operations for the Pattern Descriptor Language dialect.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_PDL_IR_PDLOPS_H_
#define AIIR_DIALECT_PDL_IR_PDLOPS_H_

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/Dialect/PDL/IR/PDLTypes.h"
#include "aiir/IR/Builders.h"
#include "aiir/IR/OpImplementation.h"
#include "aiir/IR/SymbolTable.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// PDL Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "aiir/Dialect/PDL/IR/PDLOps.h.inc"

#endif // AIIR_DIALECT_PDL_IR_PDLOPS_H_
