//===-- Optimizer/Dialect/CUFDialect.h -- CUF dialect -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://aiir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_DIALECT_CUF_CUFDIALECT_H
#define FORTRAN_OPTIMIZER_DIALECT_CUF_CUFDIALECT_H

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/IR/Dialect.h"
#include "aiir/IR/SymbolTable.h"
#include "aiir/Interfaces/CallInterfaces.h"
#include "aiir/Interfaces/FunctionInterfaces.h"
#include "aiir/Interfaces/LoopLikeInterface.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"

#include "flang/Optimizer/Dialect/CUF/CUFDialect.h.inc"

#endif // FORTRAN_OPTIMIZER_DIALECT_CUF_CUFDIALECT_H
