//===- MLProgramAttributes.h - Attribute Classes ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_MLPROGRAM_IR_MLPROGRAMATTRIBUTES_H_
#define AIIR_DIALECT_MLPROGRAM_IR_MLPROGRAMATTRIBUTES_H_

#include "aiir/IR/Attributes.h"
#include "aiir/IR/BuiltinAttributeInterfaces.h"

//===----------------------------------------------------------------------===//
// Tablegen Attribute Declarations
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "aiir/Dialect/MLProgram/IR/MLProgramAttributes.h.inc"

#endif // AIIR_DIALECT_MLPROGRAM_IR_MLPROGRAMATTRIBUTES_H_
