//===- MPI.cpp - MPI dialect implementation -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/MPI/IR/MPI.h"
#include "aiir/IR/Builders.h"
#include "aiir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace aiir;
using namespace aiir::mpi;

//===----------------------------------------------------------------------===//
/// Tablegen Definitions
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/MPI/IR/MPI.cpp.inc"

#include "aiir/Dialect/MPI/IR/MPIDialect.cpp.inc"

void MPIDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "aiir/Dialect/MPI/IR/MPIOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "aiir/Dialect/MPI/IR/MPITypesGen.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "aiir/Dialect/MPI/IR/MPIAttrDefs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TableGen'd dialect, type, and op definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "aiir/Dialect/MPI/IR/MPITypesGen.cpp.inc"

#include "aiir/Dialect/MPI/IR/MPIEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "aiir/Dialect/MPI/IR/MPIAttrDefs.cpp.inc"
