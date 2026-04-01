//===- MLProgramTypes.h - Type Classes --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_MLPROGRAM_IR_MLPROGRAMTYPES_H_
#define AIIR_DIALECT_MLPROGRAM_IR_MLPROGRAMTYPES_H_

#include "aiir/IR/Types.h"

//===----------------------------------------------------------------------===//
// Tablegen Type Declarations
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "aiir/Dialect/MLProgram/IR/MLProgramTypes.h.inc"

#endif // AIIR_DIALECT_MLPROGRAM_IR_MLPROGRAMTYPES_H_
