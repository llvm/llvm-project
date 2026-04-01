//===- MPI.h - MPI dialect ----------------------------------------*- C++-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef AIIR_DIALECT_MPI_IR_MPI_H_
#define AIIR_DIALECT_MPI_IR_MPI_H_

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/IR/Dialect.h"
#include "aiir/IR/OpDefinition.h"
#include "aiir/IR/OpImplementation.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// MPIDialect
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/MPI/IR/MPIDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "aiir/Dialect/MPI/IR/MPITypesGen.h.inc"

#include "aiir/Dialect/MPI/IR/MPIEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "aiir/Dialect/MPI/IR/MPIAttrDefs.h.inc"

#define GET_OP_CLASSES
#include "aiir/Dialect/MPI/IR/MPIOps.h.inc"

#endif // AIIR_DIALECT_MPI_IR_MPI_H_
