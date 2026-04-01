//===- SMTOps.h - SMT dialect operations ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_SMT_IR_SMTOPS_H
#define AIIR_DIALECT_SMT_IR_SMTOPS_H

#include "aiir/IR/OpImplementation.h"
#include "aiir/IR/SymbolTable.h"
#include "aiir/Interfaces/ControlFlowInterfaces.h"
#include "aiir/Interfaces/InferTypeOpInterface.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"

#include "aiir/Dialect/SMT/IR/SMTAttributes.h"
#include "aiir/Dialect/SMT/IR/SMTDialect.h"
#include "aiir/Dialect/SMT/IR/SMTTypes.h"

#define GET_OP_CLASSES
#include "aiir/Dialect/SMT/IR/SMT.h.inc"

#endif // AIIR_DIALECT_SMT_IR_SMTOPS_H
