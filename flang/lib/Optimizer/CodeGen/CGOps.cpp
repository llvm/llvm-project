//===-- CGOps.cpp -- FIR codegen operations -------------------------------===//
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

#include "CGOps.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"

/// FIR codegen dialect constructor.
fir::FIRCodeGenDialect::FIRCodeGenDialect(mlir::MLIRContext *ctx)
    : mlir::Dialect("fircg", ctx, mlir::TypeID::get<FIRCodeGenDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "flang/Optimizer/CodeGen/CGOps.cpp.inc"
      >();
}

// anchor the class vtable to this compilation unit
fir::FIRCodeGenDialect::~FIRCodeGenDialect() {
  // do nothing
}

#define GET_OP_CLASSES
#include "flang/Optimizer/CodeGen/CGOps.cpp.inc"

unsigned fir::cg::XEmboxOp::getOutRank() {
  if (slice().empty())
    return getRank();
  auto outRank = fir::SliceOp::getOutputRank(slice());
  assert(outRank >= 1);
  return outRank;
}
