//===- SMT.td - SMT dialect definition ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_SMT_IR_SMT_TD
#define AIIR_DIALECT_SMT_IR_SMT_TD

include "aiir/IR/OpBase.td"

include "aiir/Dialect/SMT/IR/SMTAttributes.td"
include "aiir/Dialect/SMT/IR/SMTDialect.td"
include "aiir/Dialect/SMT/IR/SMTTypes.td"
include "aiir/Dialect/SMT/IR/SMTOps.td"
include "aiir/Dialect/SMT/IR/SMTArrayOps.td"
include "aiir/Dialect/SMT/IR/SMTBitVectorOps.td"
include "aiir/Dialect/SMT/IR/SMTIntOps.td"

#endif // AIIR_DIALECT_SMT_IR_SMT_TD
