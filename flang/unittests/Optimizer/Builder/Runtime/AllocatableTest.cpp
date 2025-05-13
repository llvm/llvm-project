//===- AllocatableTest.cpp -- allocatable runtime builder unit tests ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/Allocatable.h"
#include "RuntimeCallTestBase.h"
#include "gtest/gtest.h"
#include "flang/Runtime/descriptor-consts.h"

using namespace Fortran::runtime;

TEST_F(RuntimeCallTest, genMoveAlloc) {
  mlir::Location loc = firBuilder->getUnknownLoc();
  mlir::Type seqTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, 10), i32Ty);
  mlir::Value from = firBuilder->create<fir::UndefOp>(loc, seqTy);
  mlir::Value to = firBuilder->create<fir::UndefOp>(loc, seqTy);
  mlir::Value errMsg = firBuilder->create<fir::UndefOp>(loc, seqTy);
  mlir::Value hasStat = firBuilder->createBool(loc, false);
  fir::runtime::genMoveAlloc(*firBuilder, loc, to, from, hasStat, errMsg);
  checkCallOpFromResultBox(to, "_FortranAMoveAlloc", 5);
}
