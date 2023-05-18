//===- AssignTest.cpp -- assignment runtime builder unit tests ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/Assign.h"
#include "RuntimeCallTestBase.h"
#include "gtest/gtest.h"

TEST_F(RuntimeCallTest, genDerivedTypeAssign) {
  auto loc = firBuilder->getUnknownLoc();
  mlir::Type seqTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, 10), i32Ty);
  mlir::Value source = firBuilder->create<fir::UndefOp>(loc, seqTy);
  mlir::Value dest = firBuilder->create<fir::UndefOp>(loc, seqTy);
  fir::runtime::genAssign(*firBuilder, loc, dest, source);
  checkCallOpFromResultBox(dest, "_FortranAAssign", 2);
}
