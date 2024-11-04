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

void testGenBesselJn(
    fir::FirOpBuilder &builder, mlir::Type realTy, llvm::StringRef fctName) {
  mlir::Location loc = builder.getUnknownLoc();
  mlir::Type i32Ty = builder.getIntegerType(32);
  mlir::Type seqTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, 10), realTy);
  mlir::Value result = builder.create<fir::UndefOp>(loc, seqTy);
  mlir::Value n1 = builder.create<fir::UndefOp>(loc, i32Ty);
  mlir::Value n2 = builder.create<fir::UndefOp>(loc, i32Ty);
  mlir::Value x = builder.create<fir::UndefOp>(loc, realTy);
  mlir::Value bn1 = builder.create<fir::UndefOp>(loc, realTy);
  mlir::Value bn2 = builder.create<fir::UndefOp>(loc, realTy);
  fir::runtime::genBesselJn(builder, loc, result, n1, n2, x, bn1, bn2);
  checkCallOpFromResultBox(result, fctName, 6);
}

TEST_F(RuntimeCallTest, genBesselJnTest) {
  testGenBesselJn(*firBuilder, f32Ty, "_FortranABesselJn_4");
  testGenBesselJn(*firBuilder, f64Ty, "_FortranABesselJn_8");
  testGenBesselJn(*firBuilder, f80Ty, "_FortranABesselJn_10");
  testGenBesselJn(*firBuilder, f128Ty, "_FortranABesselJn_16");
}

void testGenBesselJnX0(
    fir::FirOpBuilder &builder, mlir::Type realTy, llvm::StringRef fctName) {
  mlir::Location loc = builder.getUnknownLoc();
  mlir::Type i32Ty = builder.getIntegerType(32);
  mlir::Type seqTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, 10), realTy);
  mlir::Value result = builder.create<fir::UndefOp>(loc, seqTy);
  mlir::Value n1 = builder.create<fir::UndefOp>(loc, i32Ty);
  mlir::Value n2 = builder.create<fir::UndefOp>(loc, i32Ty);
  fir::runtime::genBesselJnX0(builder, loc, realTy, result, n1, n2);
  checkCallOpFromResultBox(result, fctName, 3);
}

TEST_F(RuntimeCallTest, genBesselJnX0Test) {
  testGenBesselJnX0(*firBuilder, f32Ty, "_FortranABesselJnX0_4");
  testGenBesselJnX0(*firBuilder, f64Ty, "_FortranABesselJnX0_8");
  testGenBesselJnX0(*firBuilder, f80Ty, "_FortranABesselJnX0_10");
  testGenBesselJnX0(*firBuilder, f128Ty, "_FortranABesselJnX0_16");
}

void testGenBesselYn(
    fir::FirOpBuilder &builder, mlir::Type realTy, llvm::StringRef fctName) {
  mlir::Location loc = builder.getUnknownLoc();
  mlir::Type i32Ty = builder.getIntegerType(32);
  mlir::Type seqTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, 10), realTy);
  mlir::Value result = builder.create<fir::UndefOp>(loc, seqTy);
  mlir::Value n1 = builder.create<fir::UndefOp>(loc, i32Ty);
  mlir::Value n2 = builder.create<fir::UndefOp>(loc, i32Ty);
  mlir::Value x = builder.create<fir::UndefOp>(loc, realTy);
  mlir::Value bn1 = builder.create<fir::UndefOp>(loc, realTy);
  mlir::Value bn2 = builder.create<fir::UndefOp>(loc, realTy);
  fir::runtime::genBesselYn(builder, loc, result, n1, n2, x, bn1, bn2);
  checkCallOpFromResultBox(result, fctName, 6);
}

TEST_F(RuntimeCallTest, genBesselYnTest) {
  testGenBesselYn(*firBuilder, f32Ty, "_FortranABesselYn_4");
  testGenBesselYn(*firBuilder, f64Ty, "_FortranABesselYn_8");
  testGenBesselYn(*firBuilder, f80Ty, "_FortranABesselYn_10");
  testGenBesselYn(*firBuilder, f128Ty, "_FortranABesselYn_16");
}

void testGenBesselYnX0(
    fir::FirOpBuilder &builder, mlir::Type realTy, llvm::StringRef fctName) {
  mlir::Location loc = builder.getUnknownLoc();
  mlir::Type i32Ty = builder.getIntegerType(32);
  mlir::Type seqTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, 10), realTy);
  mlir::Value result = builder.create<fir::UndefOp>(loc, seqTy);
  mlir::Value n1 = builder.create<fir::UndefOp>(loc, i32Ty);
  mlir::Value n2 = builder.create<fir::UndefOp>(loc, i32Ty);
  fir::runtime::genBesselYnX0(builder, loc, realTy, result, n1, n2);
  checkCallOpFromResultBox(result, fctName, 3);
}

TEST_F(RuntimeCallTest, genBesselYnX0Test) {
  testGenBesselYnX0(*firBuilder, f32Ty, "_FortranABesselYnX0_4");
  testGenBesselYnX0(*firBuilder, f64Ty, "_FortranABesselYnX0_8");
  testGenBesselYnX0(*firBuilder, f80Ty, "_FortranABesselYnX0_10");
  testGenBesselYnX0(*firBuilder, f128Ty, "_FortranABesselYnX0_16");
}

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

void testGenMatmul(fir::FirOpBuilder &builder, mlir::Type eleTy1,
    mlir::Type eleTy2, llvm::StringRef funcName) {
  auto loc = builder.getUnknownLoc();
  mlir::Type resultTy =
      fir::ReferenceType::get(fir::BoxType::get(builder.getNoneType()));
  mlir::Type seqTy1 =
      fir::SequenceType::get(fir::SequenceType::Shape(2, 10), eleTy1);
  mlir::Type seqTy2 =
      fir::SequenceType::get(fir::SequenceType::Shape(2, 10), eleTy2);
  mlir::Type boxTy1 = fir::BoxType::get(seqTy1);
  mlir::Type boxTy2 = fir::BoxType::get(seqTy2);
  mlir::Value result = builder.create<fir::UndefOp>(loc, resultTy);
  mlir::Value matrixA = builder.create<fir::UndefOp>(loc, boxTy1);
  mlir::Value matrixB = builder.create<fir::UndefOp>(loc, boxTy2);
  fir::runtime::genMatmul(builder, loc, result, matrixA, matrixB);
  checkCallOpFromResultBox(result, funcName, 3);
}

TEST_F(RuntimeCallTest, genMatmulTest) {
  testGenMatmul(*firBuilder, i32Ty, i16Ty, "_FortranAMatmulInteger4Integer2");
  testGenMatmul(*firBuilder, i32Ty, f64Ty, "_FortranAMatmulInteger4Real8");
  testGenMatmul(*firBuilder, i32Ty, c8Ty, "_FortranAMatmulInteger4Complex8");
  testGenMatmul(*firBuilder, f32Ty, i16Ty, "_FortranAMatmulReal4Integer2");
  testGenMatmul(*firBuilder, f32Ty, f64Ty, "_FortranAMatmulReal4Real8");
  testGenMatmul(*firBuilder, f32Ty, c8Ty, "_FortranAMatmulReal4Complex8");
  testGenMatmul(*firBuilder, c4Ty, i16Ty, "_FortranAMatmulComplex4Integer2");
  testGenMatmul(*firBuilder, c4Ty, f64Ty, "_FortranAMatmulComplex4Real8");
  testGenMatmul(*firBuilder, c4Ty, c8Ty, "_FortranAMatmulComplex4Complex8");
  testGenMatmul(*firBuilder, f80Ty, f128Ty, "_FortranAMatmulReal10Real16");
  testGenMatmul(*firBuilder, f80Ty, i128Ty, "_FortranAMatmulReal10Integer16");
  testGenMatmul(*firBuilder, f128Ty, i128Ty, "_FortranAMatmulReal16Integer16");
  testGenMatmul(
      *firBuilder, logical1Ty, logical2Ty, "_FortranAMatmulLogical1Logical2");
  testGenMatmul(
      *firBuilder, logical4Ty, logical8Ty, "_FortranAMatmulLogical4Logical8");
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
