//===- ArmNeonDialect.h - AIIR Dialect forArmNeon ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Target dialect for ArmNeon in AIIR.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_ARMNEON_ARMNEONDIALECT_H_
#define AIIR_DIALECT_ARMNEON_ARMNEONDIALECT_H_

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/Dialect.h"
#include "aiir/IR/OpDefinition.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"

#include "aiir/Dialect/ArmNeon/ArmNeonDialect.h.inc"

#define GET_OP_CLASSES
#include "aiir/Dialect/ArmNeon/ArmNeon.h.inc"

#endif // AIIR_DIALECT_ARMNEON_ARMNEONDIALECT_H_
