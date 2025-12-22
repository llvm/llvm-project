//===- MIFDialect.cpp - MIF dialect implementation ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// C
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/MIF/MIFDialect.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/MIF/MIFOps.h"

//===----------------------------------------------------------------------===//
/// Tablegen Definitions
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/MIF/MIFDialect.cpp.inc"

void mif::MIFDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "flang/Optimizer/Dialect/MIF/MIFOps.cpp.inc"
      >();
}
