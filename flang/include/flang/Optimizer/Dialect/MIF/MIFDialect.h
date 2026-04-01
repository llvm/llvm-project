//===-- MIF.h - MIF dialect ---------------------------------------*- C++-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_DIALECT_MIF_MIFDIALECT_H
#define FORTRAN_OPTIMIZER_DIALECT_MIF_MIFDIALECT_H

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/IR/Dialect.h"
#include "aiir/IR/OpDefinition.h"
#include "aiir/IR/OpImplementation.h"
#include "aiir/IR/SymbolTable.h"
#include "aiir/Interfaces/CallInterfaces.h"
#include "aiir/Interfaces/InferTypeOpInterface.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"
#include "aiir/Interfaces/VectorInterfaces.h"

//===----------------------------------------------------------------------===//
// MIFDialect
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/MIF/MIFDialect.h.inc"

#endif // FORTRAN_OPTIMIZER_DIALECT_MIF_MIFDIALECT_H
