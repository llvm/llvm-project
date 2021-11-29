//===- TransformationalTest.cpp -- Transformational intrinsic generation --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/Transformational.h"
#include "RuntimeCallTestBase.h"
#include "gtest/gtest.h"

TEST_F(RuntimeCallTest, genCshiftTest) {
  auto loc = firBuilder->getUnknownLoc();
  mlir::Type seqTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, 10), i32Ty);
  mlir::Value result = firBuilder->create<fir::UndefOp>(loc, seqTy);
  mlir::Value array = firBuilder->create<fir::UndefOp>(loc, seqTy);
  mlir::Value shift = firBuilder->create<fir::UndefOp>(loc, seqTy);
  mlir::Value dim = firBuilder->create<fir::UndefOp>(loc, seqTy);
  fir::runtime::genCshift(*firBuilder, loc, result, array, shift, dim);
  checkCallOpFromResultBox(result, "_FortranACshift", 4);
}

TEST_F(RuntimeCallTest, genCshiftVectorTest) {
  auto loc = firBuilder->getUnknownLoc();
  mlir::Type seqTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, 10), i32Ty);
  mlir::Value result = firBuilder->create<fir::UndefOp>(loc, seqTy);
  mlir::Value array = firBuilder->create<fir::UndefOp>(loc, seqTy);
  mlir::Value shift = firBuilder->create<fir::UndefOp>(loc, seqTy);
  fir::runtime::genCshiftVector(*firBuilder, loc, result, array, shift);
  checkCallOpFromResultBox(result, "_FortranACshiftVector", 3);
}

TEST_F(RuntimeCallTest, genEoshiftTest) {
  auto loc = firBuilder->getUnknownLoc();
  mlir::Type seqTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, 10), i32Ty);
  mlir::Value result = firBuilder->create<fir::UndefOp>(loc, seqTy);
  mlir::Value array = firBuilder->create<fir::UndefOp>(loc, seqTy);
  mlir::Value shift = firBuilder->create<fir::UndefOp>(loc, seqTy);
  mlir::Value bound = firBuilder->create<fir::UndefOp>(loc, seqTy);
  mlir::Value dim = firBuilder->create<fir::UndefOp>(loc, seqTy);
  fir::runtime::genEoshift(*firBuilder, loc, result, array, shift, bound, dim);
  checkCallOpFromResultBox(result, "_FortranAEoshift", 5);
}

TEST_F(RuntimeCallTest, genEoshiftVectorTest) {
  auto loc = firBuilder->getUnknownLoc();
  mlir::Type seqTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, 10), i32Ty);
  mlir::Value result = firBuilder->create<fir::UndefOp>(loc, seqTy);
  mlir::Value array = firBuilder->create<fir::UndefOp>(loc, seqTy);
  mlir::Value shift = firBuilder->create<fir::UndefOp>(loc, seqTy);
  mlir::Value bound = firBuilder->create<fir::UndefOp>(loc, seqTy);
  fir::runtime::genEoshiftVector(*firBuilder, loc, result, array, shift, bound);
  checkCallOpFromResultBox(result, "_FortranAEoshiftVector", 4);
}

TEST_F(RuntimeCallTest, genMatmulTest) {
  auto loc = firBuilder->getUnknownLoc();
  mlir::Type seqTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, 10), i32Ty);
  mlir::Value result = firBuilder->create<fir::UndefOp>(loc, seqTy);
  mlir::Value matrixA = firBuilder->create<fir::UndefOp>(loc, seqTy);
  mlir::Value matrixB = firBuilder->create<fir::UndefOp>(loc, seqTy);
  fir::runtime::genMatmul(*firBuilder, loc, matrixA, matrixB, result);
  checkCallOpFromResultBox(result, "_FortranAMatmul", 3);
}

TEST_F(RuntimeCallTest, genPackTest) {
  auto loc = firBuilder->getUnknownLoc();
  mlir::Type seqTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, 10), i32Ty);
  mlir::Value result = firBuilder->create<fir::UndefOp>(loc, seqTy);
  mlir::Value array = firBuilder->create<fir::UndefOp>(loc, seqTy);
  mlir::Value mask = firBuilder->create<fir::UndefOp>(loc, seqTy);
  mlir::Value vector = firBuilder->create<fir::UndefOp>(loc, seqTy);
  fir::runtime::genPack(*firBuilder, loc, result, array, mask, vector);
  checkCallOpFromResultBox(result, "_FortranAPack", 4);
}

TEST_F(RuntimeCallTest, genReshapeTest) {
  auto loc = firBuilder->getUnknownLoc();
  mlir::Type seqTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, 10), i32Ty);
  mlir::Value result = firBuilder->create<fir::UndefOp>(loc, seqTy);
  mlir::Value source = firBuilder->create<fir::UndefOp>(loc, seqTy);
  mlir::Value shape = firBuilder->create<fir::UndefOp>(loc, seqTy);
  mlir::Value pad = firBuilder->create<fir::UndefOp>(loc, seqTy);
  mlir::Value order = firBuilder->create<fir::UndefOp>(loc, seqTy);
  fir::runtime::genReshape(*firBuilder, loc, result, source, shape, pad, order);
  checkCallOpFromResultBox(result, "_FortranAReshape", 5);
}

TEST_F(RuntimeCallTest, genSpreadTest) {
  auto loc = firBuilder->getUnknownLoc();
  mlir::Type seqTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, 10), i32Ty);
  mlir::Value result = firBuilder->create<fir::UndefOp>(loc, seqTy);
  mlir::Value source = firBuilder->create<fir::UndefOp>(loc, seqTy);
  mlir::Value dim = firBuilder->create<fir::UndefOp>(loc, seqTy);
  mlir::Value ncopies = firBuilder->create<fir::UndefOp>(loc, seqTy);
  fir::runtime::genSpread(*firBuilder, loc, result, source, dim, ncopies);
  checkCallOpFromResultBox(result, "_FortranASpread", 4);
}

TEST_F(RuntimeCallTest, genTransposeTest) {
  auto loc = firBuilder->getUnknownLoc();
  mlir::Type seqTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, 10), i32Ty);
  mlir::Value result = firBuilder->create<fir::UndefOp>(loc, seqTy);
  mlir::Value source = firBuilder->create<fir::UndefOp>(loc, seqTy);
  fir::runtime::genTranspose(*firBuilder, loc, result, source);
  checkCallOpFromResultBox(result, "_FortranATranspose", 2);
}

TEST_F(RuntimeCallTest, genUnpack) {
  auto loc = firBuilder->getUnknownLoc();
  mlir::Type seqTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, 10), i32Ty);
  mlir::Value result = firBuilder->create<fir::UndefOp>(loc, seqTy);
  mlir::Value vector = firBuilder->create<fir::UndefOp>(loc, seqTy);
  mlir::Value mask = firBuilder->create<fir::UndefOp>(loc, seqTy);
  mlir::Value field = firBuilder->create<fir::UndefOp>(loc, seqTy);
  fir::runtime::genUnpack(*firBuilder, loc, result, vector, mask, field);
  checkCallOpFromResultBox(result, "_FortranAUnpack", 4);
}
