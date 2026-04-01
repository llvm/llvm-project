//===- Complex.h - Complex dialect --------------------------------*- C++-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_COMPLEX_IR_COMPLEX_H_
#define AIIR_DIALECT_COMPLEX_IR_COMPLEX_H_

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/Dialect/Arith/IR/Arith.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/OpImplementation.h"
#include "aiir/Interfaces/InferTypeOpInterface.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// Complex Dialect
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Complex/IR/ComplexOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Complex Dialect Enums
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Complex/IR/ComplexEnums.h.inc"

//===----------------------------------------------------------------------===//
// Complex Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "aiir/Dialect/Complex/IR/ComplexOps.h.inc"

#define GET_ATTRDEF_CLASSES
#include "aiir/Dialect/Complex/IR/ComplexAttributes.h.inc"

#endif // AIIR_DIALECT_COMPLEX_IR_COMPLEX_H_
