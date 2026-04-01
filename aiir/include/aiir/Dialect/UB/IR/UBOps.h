//===- UBOps.h - UB Dialect Operations ------------------------*--- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_UB_IR_OPS_H
#define AIIR_DIALECT_UB_IR_OPS_H

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/IR/Dialect.h"
#include "aiir/IR/OpImplementation.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"

#include "aiir/Dialect/UB/IR/UBOpsInterfaces.h.inc"

#define GET_ATTRDEF_CLASSES
#include "aiir/Dialect/UB/IR/UBOpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "aiir/Dialect/UB/IR/UBOps.h.inc"

#include "aiir/Dialect/UB/IR/UBOpsDialect.h.inc"

#endif // AIIR_DIALECT_UB_IR_OPS_H
