//===- DerivedTest.cpp -- Derived type runtime builder unit tests ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/Derived.h"
#include "RuntimeCallTestBase.h"
#include "gtest/gtest.h"

TEST_F(RuntimeCallTest, genDerivedTypeInitialize) {
  auto loc = firBuilder->getUnknownLoc();
  mlir::Type seqTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, 10), i32Ty);
  mlir::Value box = firBuilder->create<fir::UndefOp>(loc, seqTy);
  fir::runtime::genDerivedTypeInitialize(*firBuilder, loc, box);
  checkCallOpFromResultBox(box, "_FortranAInitialize", 1);
}

TEST_F(RuntimeCallTest, genDerivedTypeDestroy) {
  auto loc = firBuilder->getUnknownLoc();
  mlir::Type seqTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, 10), i32Ty);
  mlir::Value box = firBuilder->create<fir::UndefOp>(loc, seqTy);
  fir::runtime::genDerivedTypeDestroy(*firBuilder, loc, box);
  checkCallOpFromResultBox(box, "_FortranADestroy", 1, /*addLocArg=*/false);
}
