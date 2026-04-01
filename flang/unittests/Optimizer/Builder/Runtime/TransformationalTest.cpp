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
    fir::FirOpBuilder &builder, aiir::Type realTy, llvm::StringRef fctName) {
  aiir::Location loc = builder.getUnknownLoc();
  aiir::Type i32Ty = builder.getIntegerType(32);
  aiir::Type seqTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, 10), realTy);
  aiir::Value result = fir::UndefOp::create(builder, loc, seqTy);
  aiir::Value n1 = fir::UndefOp::create(builder, loc, i32Ty);
  aiir::Value n2 = fir::UndefOp::create(builder, loc, i32Ty);
  aiir::Value x = fir::UndefOp::create(builder, loc, realTy);
  aiir::Value bn1 = fir::UndefOp::create(builder, loc, realTy);
  aiir::Value bn2 = fir::UndefOp::create(builder, loc, realTy);
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
    fir::FirOpBuilder &builder, aiir::Type realTy, llvm::StringRef fctName) {
  aiir::Location loc = builder.getUnknownLoc();
  aiir::Type i32Ty = builder.getIntegerType(32);
  aiir::Type seqTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, 10), realTy);
  aiir::Value result = fir::UndefOp::create(builder, loc, seqTy);
  aiir::Value n1 = fir::UndefOp::create(builder, loc, i32Ty);
  aiir::Value n2 = fir::UndefOp::create(builder, loc, i32Ty);
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
    fir::FirOpBuilder &builder, aiir::Type realTy, llvm::StringRef fctName) {
  aiir::Location loc = builder.getUnknownLoc();
  aiir::Type i32Ty = builder.getIntegerType(32);
  aiir::Type seqTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, 10), realTy);
  aiir::Value result = fir::UndefOp::create(builder, loc, seqTy);
  aiir::Value n1 = fir::UndefOp::create(builder, loc, i32Ty);
  aiir::Value n2 = fir::UndefOp::create(builder, loc, i32Ty);
  aiir::Value x = fir::UndefOp::create(builder, loc, realTy);
  aiir::Value bn1 = fir::UndefOp::create(builder, loc, realTy);
  aiir::Value bn2 = fir::UndefOp::create(builder, loc, realTy);
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
    fir::FirOpBuilder &builder, aiir::Type realTy, llvm::StringRef fctName) {
  aiir::Location loc = builder.getUnknownLoc();
  aiir::Type i32Ty = builder.getIntegerType(32);
  aiir::Type seqTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, 10), realTy);
  aiir::Value result = fir::UndefOp::create(builder, loc, seqTy);
  aiir::Value n1 = fir::UndefOp::create(builder, loc, i32Ty);
  aiir::Value n2 = fir::UndefOp::create(builder, loc, i32Ty);
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
  aiir::Type seqTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, 10), i32Ty);
  aiir::Value result = fir::UndefOp::create(*firBuilder, loc, seqTy);
  aiir::Value array = fir::UndefOp::create(*firBuilder, loc, seqTy);
  aiir::Value shift = fir::UndefOp::create(*firBuilder, loc, seqTy);
  aiir::Value dim = fir::UndefOp::create(*firBuilder, loc, seqTy);
  fir::runtime::genCshift(*firBuilder, loc, result, array, shift, dim);
  checkCallOpFromResultBox(result, "_FortranACshift", 4);
}

TEST_F(RuntimeCallTest, genCshiftVectorTest) {
  auto loc = firBuilder->getUnknownLoc();
  aiir::Type seqTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, 10), i32Ty);
  aiir::Value result = fir::UndefOp::create(*firBuilder, loc, seqTy);
  aiir::Value array = fir::UndefOp::create(*firBuilder, loc, seqTy);
  aiir::Value shift = fir::UndefOp::create(*firBuilder, loc, seqTy);
  fir::runtime::genCshiftVector(*firBuilder, loc, result, array, shift);
  checkCallOpFromResultBox(result, "_FortranACshiftVector", 3);
}

TEST_F(RuntimeCallTest, genEoshiftTest) {
  auto loc = firBuilder->getUnknownLoc();
  aiir::Type seqTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, 10), i32Ty);
  aiir::Value result = fir::UndefOp::create(*firBuilder, loc, seqTy);
  aiir::Value array = fir::UndefOp::create(*firBuilder, loc, seqTy);
  aiir::Value shift = fir::UndefOp::create(*firBuilder, loc, seqTy);
  aiir::Value bound = fir::UndefOp::create(*firBuilder, loc, seqTy);
  aiir::Value dim = fir::UndefOp::create(*firBuilder, loc, seqTy);
  fir::runtime::genEoshift(*firBuilder, loc, result, array, shift, bound, dim);
  checkCallOpFromResultBox(result, "_FortranAEoshift", 5);
}

TEST_F(RuntimeCallTest, genEoshiftVectorTest) {
  auto loc = firBuilder->getUnknownLoc();
  aiir::Type seqTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, 10), i32Ty);
  aiir::Value result = fir::UndefOp::create(*firBuilder, loc, seqTy);
  aiir::Value array = fir::UndefOp::create(*firBuilder, loc, seqTy);
  aiir::Value shift = fir::UndefOp::create(*firBuilder, loc, seqTy);
  aiir::Value bound = fir::UndefOp::create(*firBuilder, loc, seqTy);
  fir::runtime::genEoshiftVector(*firBuilder, loc, result, array, shift, bound);
  checkCallOpFromResultBox(result, "_FortranAEoshiftVector", 4);
}

void testGenMatmul(fir::FirOpBuilder &builder, aiir::Type eleTy1,
    aiir::Type eleTy2, llvm::StringRef funcName) {
  auto loc = builder.getUnknownLoc();
  aiir::Type resultTy =
      fir::ReferenceType::get(fir::BoxType::get(builder.getNoneType()));
  aiir::Type seqTy1 =
      fir::SequenceType::get(fir::SequenceType::Shape(2, 10), eleTy1);
  aiir::Type seqTy2 =
      fir::SequenceType::get(fir::SequenceType::Shape(2, 10), eleTy2);
  aiir::Type boxTy1 = fir::BoxType::get(seqTy1);
  aiir::Type boxTy2 = fir::BoxType::get(seqTy2);
  aiir::Value result = fir::UndefOp::create(builder, loc, resultTy);
  aiir::Value matrixA = fir::UndefOp::create(builder, loc, boxTy1);
  aiir::Value matrixB = fir::UndefOp::create(builder, loc, boxTy2);
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
  aiir::Type seqTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, 10), i32Ty);
  aiir::Value result = fir::UndefOp::create(*firBuilder, loc, seqTy);
  aiir::Value array = fir::UndefOp::create(*firBuilder, loc, seqTy);
  aiir::Value mask = fir::UndefOp::create(*firBuilder, loc, seqTy);
  aiir::Value vector = fir::UndefOp::create(*firBuilder, loc, seqTy);
  fir::runtime::genPack(*firBuilder, loc, result, array, mask, vector);
  checkCallOpFromResultBox(result, "_FortranAPack", 4);
}

TEST_F(RuntimeCallTest, genReshapeTest) {
  auto loc = firBuilder->getUnknownLoc();
  aiir::Type seqTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, 10), i32Ty);
  aiir::Value result = fir::UndefOp::create(*firBuilder, loc, seqTy);
  aiir::Value source = fir::UndefOp::create(*firBuilder, loc, seqTy);
  aiir::Value shape = fir::UndefOp::create(*firBuilder, loc, seqTy);
  aiir::Value pad = fir::UndefOp::create(*firBuilder, loc, seqTy);
  aiir::Value order = fir::UndefOp::create(*firBuilder, loc, seqTy);
  fir::runtime::genReshape(*firBuilder, loc, result, source, shape, pad, order);
  checkCallOpFromResultBox(result, "_FortranAReshape", 5);
}

TEST_F(RuntimeCallTest, genSpreadTest) {
  auto loc = firBuilder->getUnknownLoc();
  aiir::Type seqTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, 10), i32Ty);
  aiir::Value result = fir::UndefOp::create(*firBuilder, loc, seqTy);
  aiir::Value source = fir::UndefOp::create(*firBuilder, loc, seqTy);
  aiir::Value dim = fir::UndefOp::create(*firBuilder, loc, seqTy);
  aiir::Value ncopies = fir::UndefOp::create(*firBuilder, loc, seqTy);
  fir::runtime::genSpread(*firBuilder, loc, result, source, dim, ncopies);
  checkCallOpFromResultBox(result, "_FortranASpread", 4);
}

TEST_F(RuntimeCallTest, genTransposeTest) {
  auto loc = firBuilder->getUnknownLoc();
  aiir::Type seqTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, 10), i32Ty);
  aiir::Value result = fir::UndefOp::create(*firBuilder, loc, seqTy);
  aiir::Value source = fir::UndefOp::create(*firBuilder, loc, seqTy);
  fir::runtime::genTranspose(*firBuilder, loc, result, source);
  checkCallOpFromResultBox(result, "_FortranATranspose", 2);
}

TEST_F(RuntimeCallTest, genUnpack) {
  auto loc = firBuilder->getUnknownLoc();
  aiir::Type seqTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, 10), i32Ty);
  aiir::Value result = fir::UndefOp::create(*firBuilder, loc, seqTy);
  aiir::Value vector = fir::UndefOp::create(*firBuilder, loc, seqTy);
  aiir::Value mask = fir::UndefOp::create(*firBuilder, loc, seqTy);
  aiir::Value field = fir::UndefOp::create(*firBuilder, loc, seqTy);
  fir::runtime::genUnpack(*firBuilder, loc, result, vector, mask, field);
  checkCallOpFromResultBox(result, "_FortranAUnpack", 4);
}
