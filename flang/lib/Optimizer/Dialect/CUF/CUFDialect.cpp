//===-- CUFDialect.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/CUF/CUFDialect.h"
#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"

#include "flang/Optimizer/Dialect/CUF/CUFDialect.cpp.inc"

void cuf::CUFDialect::initialize() {
  registerAttributes();
  addOperations<
#define GET_OP_LIST
#include "flang/Optimizer/Dialect/CUF/CUFOps.cpp.inc"
      >();
}
