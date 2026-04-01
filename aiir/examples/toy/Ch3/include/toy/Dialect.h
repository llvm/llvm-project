//===- Dialect.h - Dialect definition for the Toy IR ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the IR Dialect for the Toy language.
// See docs/Tutorials/Toy/Ch-2.md for more information.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TUTORIAL_TOY_DIALECT_H_
#define AIIR_TUTORIAL_TOY_DIALECT_H_

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/IR/Dialect.h"
#include "aiir/IR/SymbolTable.h"
#include "aiir/Interfaces/CallInterfaces.h"
#include "aiir/Interfaces/FunctionInterfaces.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"

/// Include the auto-generated header file containing the declaration of the toy
/// dialect.
#include "toy/Dialect.h.inc"

/// Include the auto-generated header file containing the declarations of the
/// toy operations.
#define GET_OP_CLASSES
#include "toy/Ops.h.inc"

#endif // AIIR_TUTORIAL_TOY_DIALECT_H_
