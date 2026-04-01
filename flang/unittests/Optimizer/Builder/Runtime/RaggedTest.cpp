//===- RaggedTest.cpp -- Ragged array runtime function builder unit tests -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/Ragged.h"
#include "RuntimeCallTestBase.h"
#include "gtest/gtest.h"

TEST_F(RuntimeCallTest, genRaggedArrayAllocateTest) {
  auto loc = firBuilder->getUnknownLoc();
  aiir::TupleType headerTy =
      fir::factory::getRaggedArrayHeaderType(*firBuilder);
  aiir::Value header = fir::UndefOp::create(*firBuilder, loc, headerTy);
  aiir::Value eleSize = firBuilder->createIntegerConstant(loc, i32Ty, 1);
  aiir::Value extent = firBuilder->createIntegerConstant(loc, i32Ty, 1);
  // Use a dummy header just to test the correctness of the generated call.
  fir::runtime::genRaggedArrayAllocate(
      loc, *firBuilder, header, false, eleSize, {extent});
  checkCallOpFromResultBox(
      eleSize, "_FortranARaggedArrayAllocate", 5, /*addLocArgs=*/false);
}

TEST_F(RuntimeCallTest, genRaggedArrayDeallocateTest) {
  auto loc = firBuilder->getUnknownLoc();
  aiir::TupleType headerTy =
      fir::factory::getRaggedArrayHeaderType(*firBuilder);
  // Use a dummy header just to test the correctness of the generated call.
  aiir::Value header = fir::UndefOp::create(*firBuilder, loc, headerTy);
  fir::runtime::genRaggedArrayDeallocate(loc, *firBuilder, header);
  checkCallOpFromResultBox(
      header, "_FortranARaggedArrayDeallocate", 1, /*addLocArgs=*/false);
}
