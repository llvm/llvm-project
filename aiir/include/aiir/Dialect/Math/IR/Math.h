//===- Math.h - Math dialect --------------------------------------*- C++-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_MATH_IR_MATH_H_
#define AIIR_DIALECT_MATH_IR_MATH_H_

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/Dialect/Arith/IR/Arith.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/Dialect.h"
#include "aiir/IR/OpDefinition.h"
#include "aiir/IR/OpImplementation.h"
#include "aiir/Interfaces/InferTypeOpInterface.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"
#include "aiir/Interfaces/VectorInterfaces.h"

//===----------------------------------------------------------------------===//
// Math Dialect
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Math/IR/MathOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Math Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "aiir/Dialect/Math/IR/MathOps.h.inc"

#endif // AIIR_DIALECT_MATH_IR_MATH_H_
