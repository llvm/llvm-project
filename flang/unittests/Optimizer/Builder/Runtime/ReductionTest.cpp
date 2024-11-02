//===- ReductionTest.cpp -- Reduction runtime builder unit tests ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/Reduction.h"
#include "RuntimeCallTestBase.h"
#include "gtest/gtest.h"

TEST_F(RuntimeCallTest, genAllTest) {
  mlir::Location loc = firBuilder->getUnknownLoc();
  mlir::Value undef = firBuilder->create<fir::UndefOp>(loc, seqTy10);
  mlir::Value dim = firBuilder->createIntegerConstant(loc, i32Ty, 1);
  mlir::Value all = fir::runtime::genAll(*firBuilder, loc, undef, dim);
  checkCallOp(all.getDefiningOp(), "_FortranAAll", 2);
}

TEST_F(RuntimeCallTest, genAllDescriptorTest) {
  mlir::Location loc = firBuilder->getUnknownLoc();
  mlir::Value result = firBuilder->create<fir::UndefOp>(loc, seqTy10);
  mlir::Value mask = firBuilder->create<fir::UndefOp>(loc, seqTy10);
  mlir::Value dim = firBuilder->createIntegerConstant(loc, i32Ty, 1);
  fir::runtime::genAllDescriptor(*firBuilder, loc, result, mask, dim);
  checkCallOpFromResultBox(result, "_FortranAAllDim", 3);
}

TEST_F(RuntimeCallTest, genAnyTest) {
  mlir::Location loc = firBuilder->getUnknownLoc();
  mlir::Value undef = firBuilder->create<fir::UndefOp>(loc, seqTy10);
  mlir::Value dim = firBuilder->createIntegerConstant(loc, i32Ty, 1);
  mlir::Value any = fir::runtime::genAny(*firBuilder, loc, undef, dim);
  checkCallOp(any.getDefiningOp(), "_FortranAAny", 2);
}

TEST_F(RuntimeCallTest, genAnyDescriptorTest) {
  mlir::Location loc = firBuilder->getUnknownLoc();
  mlir::Value result = firBuilder->create<fir::UndefOp>(loc, seqTy10);
  mlir::Value mask = firBuilder->create<fir::UndefOp>(loc, seqTy10);
  mlir::Value dim = firBuilder->createIntegerConstant(loc, i32Ty, 1);
  fir::runtime::genAnyDescriptor(*firBuilder, loc, result, mask, dim);
  checkCallOpFromResultBox(result, "_FortranAAnyDim", 3);
}

TEST_F(RuntimeCallTest, genCountTest) {
  mlir::Location loc = firBuilder->getUnknownLoc();
  mlir::Value undef = firBuilder->create<fir::UndefOp>(loc, seqTy10);
  mlir::Value dim = firBuilder->createIntegerConstant(loc, i32Ty, 1);
  mlir::Value count = fir::runtime::genCount(*firBuilder, loc, undef, dim);
  checkCallOp(count.getDefiningOp(), "_FortranACount", 2);
}

TEST_F(RuntimeCallTest, genCountDimTest) {
  mlir::Location loc = firBuilder->getUnknownLoc();
  mlir::Value result = firBuilder->create<fir::UndefOp>(loc, seqTy10);
  mlir::Value mask = firBuilder->create<fir::UndefOp>(loc, seqTy10);
  mlir::Value dim = firBuilder->createIntegerConstant(loc, i32Ty, 1);
  mlir::Value kind = firBuilder->createIntegerConstant(loc, i32Ty, 1);
  fir::runtime::genCountDim(*firBuilder, loc, result, mask, dim, kind);
  checkCallOpFromResultBox(result, "_FortranACountDim", 4);
}

void testGenMaxVal(
    fir::FirOpBuilder &builder, mlir::Type eleTy, llvm::StringRef fctName) {
  mlir::Location loc = builder.getUnknownLoc();
  mlir::Type seqTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, 10), eleTy);
  mlir::Type refSeqTy = fir::ReferenceType::get(seqTy);
  mlir::Value undef = builder.create<fir::UndefOp>(loc, refSeqTy);
  mlir::Value mask = builder.create<fir::UndefOp>(loc, seqTy);
  mlir::Value max = fir::runtime::genMaxval(builder, loc, undef, mask);
  checkCallOp(max.getDefiningOp(), fctName, 3);
}

TEST_F(RuntimeCallTest, genMaxValTest) {
  testGenMaxVal(*firBuilder, f32Ty, "_FortranAMaxvalReal4");
  testGenMaxVal(*firBuilder, f64Ty, "_FortranAMaxvalReal8");
  testGenMaxVal(*firBuilder, f80Ty, "_FortranAMaxvalReal10");
  testGenMaxVal(*firBuilder, f128Ty, "_FortranAMaxvalReal16");

  testGenMaxVal(*firBuilder, i8Ty, "_FortranAMaxvalInteger1");
  testGenMaxVal(*firBuilder, i16Ty, "_FortranAMaxvalInteger2");
  testGenMaxVal(*firBuilder, i32Ty, "_FortranAMaxvalInteger4");
  testGenMaxVal(*firBuilder, i64Ty, "_FortranAMaxvalInteger8");
  testGenMaxVal(*firBuilder, i128Ty, "_FortranAMaxvalInteger16");
}

void testGenMinVal(
    fir::FirOpBuilder &builder, mlir::Type eleTy, llvm::StringRef fctName) {
  mlir::Location loc = builder.getUnknownLoc();
  mlir::Type seqTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, 10), eleTy);
  mlir::Type refSeqTy = fir::ReferenceType::get(seqTy);
  mlir::Value undef = builder.create<fir::UndefOp>(loc, refSeqTy);
  mlir::Value mask = builder.create<fir::UndefOp>(loc, seqTy);
  mlir::Value min = fir::runtime::genMinval(builder, loc, undef, mask);
  checkCallOp(min.getDefiningOp(), fctName, 3);
}

TEST_F(RuntimeCallTest, genMinValTest) {
  testGenMinVal(*firBuilder, f32Ty, "_FortranAMinvalReal4");
  testGenMinVal(*firBuilder, f64Ty, "_FortranAMinvalReal8");
  testGenMinVal(*firBuilder, f80Ty, "_FortranAMinvalReal10");
  testGenMinVal(*firBuilder, f128Ty, "_FortranAMinvalReal16");

  testGenMinVal(*firBuilder, i8Ty, "_FortranAMinvalInteger1");
  testGenMinVal(*firBuilder, i16Ty, "_FortranAMinvalInteger2");
  testGenMinVal(*firBuilder, i32Ty, "_FortranAMinvalInteger4");
  testGenMinVal(*firBuilder, i64Ty, "_FortranAMinvalInteger8");
  testGenMinVal(*firBuilder, i128Ty, "_FortranAMinvalInteger16");
}

TEST_F(RuntimeCallTest, genParityTest) {
  mlir::Location loc = firBuilder->getUnknownLoc();
  mlir::Value undef = firBuilder->create<fir::UndefOp>(loc, seqTy10);
  mlir::Value dim = firBuilder->createIntegerConstant(loc, i32Ty, 1);
  mlir::Value parity = fir::runtime::genParity(*firBuilder, loc, undef, dim);
  checkCallOp(parity.getDefiningOp(), "_FortranAParity", 2);
}

TEST_F(RuntimeCallTest, genParityDescriptorTest) {
  mlir::Location loc = firBuilder->getUnknownLoc();
  mlir::Value result = firBuilder->create<fir::UndefOp>(loc, seqTy10);
  mlir::Value mask = firBuilder->create<fir::UndefOp>(loc, seqTy10);
  mlir::Value dim = firBuilder->createIntegerConstant(loc, i32Ty, 1);
  fir::runtime::genParityDescriptor(*firBuilder, loc, result, mask, dim);
  checkCallOpFromResultBox(result, "_FortranAParityDim", 3);
}

void testGenSum(
    fir::FirOpBuilder &builder, mlir::Type eleTy, llvm::StringRef fctName) {
  mlir::Location loc = builder.getUnknownLoc();
  mlir::Type seqTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, 10), eleTy);
  mlir::Type refSeqTy = fir::ReferenceType::get(seqTy);
  mlir::Value undef = builder.create<fir::UndefOp>(loc, refSeqTy);
  mlir::Value mask = builder.create<fir::UndefOp>(loc, seqTy);
  mlir::Value result = builder.create<fir::UndefOp>(loc, seqTy);
  mlir::Value sum = fir::runtime::genSum(builder, loc, undef, mask, result);
  if (fir::isa_complex(eleTy))
    checkCallOpFromResultBox(result, fctName, 4);
  else
    checkCallOp(sum.getDefiningOp(), fctName, 3);
}

TEST_F(RuntimeCallTest, genSumTest) {
  testGenSum(*firBuilder, f32Ty, "_FortranASumReal4");
  testGenSum(*firBuilder, f64Ty, "_FortranASumReal8");
  testGenSum(*firBuilder, f80Ty, "_FortranASumReal10");
  testGenSum(*firBuilder, f128Ty, "_FortranASumReal16");
  testGenSum(*firBuilder, i8Ty, "_FortranASumInteger1");
  testGenSum(*firBuilder, i16Ty, "_FortranASumInteger2");
  testGenSum(*firBuilder, i32Ty, "_FortranASumInteger4");
  testGenSum(*firBuilder, i64Ty, "_FortranASumInteger8");
  testGenSum(*firBuilder, i128Ty, "_FortranASumInteger16");
  testGenSum(*firBuilder, c4Ty, "_FortranACppSumComplex4");
  testGenSum(*firBuilder, c8Ty, "_FortranACppSumComplex8");
  testGenSum(*firBuilder, c10Ty, "_FortranACppSumComplex10");
  testGenSum(*firBuilder, c16Ty, "_FortranACppSumComplex16");
}

void testGenProduct(
    fir::FirOpBuilder &builder, mlir::Type eleTy, llvm::StringRef fctName) {
  mlir::Location loc = builder.getUnknownLoc();
  mlir::Type seqTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, 10), eleTy);
  mlir::Type refSeqTy = fir::ReferenceType::get(seqTy);
  mlir::Value undef = builder.create<fir::UndefOp>(loc, refSeqTy);
  mlir::Value mask = builder.create<fir::UndefOp>(loc, seqTy);
  mlir::Value result = builder.create<fir::UndefOp>(loc, seqTy);
  mlir::Value prod =
      fir::runtime::genProduct(builder, loc, undef, mask, result);
  if (fir::isa_complex(eleTy))
    checkCallOpFromResultBox(result, fctName, 4);
  else
    checkCallOp(prod.getDefiningOp(), fctName, 3);
}

TEST_F(RuntimeCallTest, genProduct) {
  testGenProduct(*firBuilder, f32Ty, "_FortranAProductReal4");
  testGenProduct(*firBuilder, f64Ty, "_FortranAProductReal8");
  testGenProduct(*firBuilder, f80Ty, "_FortranAProductReal10");
  testGenProduct(*firBuilder, f128Ty, "_FortranAProductReal16");
  testGenProduct(*firBuilder, i8Ty, "_FortranAProductInteger1");
  testGenProduct(*firBuilder, i16Ty, "_FortranAProductInteger2");
  testGenProduct(*firBuilder, i32Ty, "_FortranAProductInteger4");
  testGenProduct(*firBuilder, i64Ty, "_FortranAProductInteger8");
  testGenProduct(*firBuilder, i128Ty, "_FortranAProductInteger16");
  testGenProduct(*firBuilder, c4Ty, "_FortranACppProductComplex4");
  testGenProduct(*firBuilder, c8Ty, "_FortranACppProductComplex8");
  testGenProduct(*firBuilder, c10Ty, "_FortranACppProductComplex10");
  testGenProduct(*firBuilder, c16Ty, "_FortranACppProductComplex16");
}

void testGenDotProduct(
    fir::FirOpBuilder &builder, mlir::Type eleTy, llvm::StringRef fctName) {
  mlir::Location loc = builder.getUnknownLoc();
  mlir::Type seqTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, 10), eleTy);
  mlir::Type refSeqTy = fir::ReferenceType::get(seqTy);
  mlir::Value a = builder.create<fir::UndefOp>(loc, refSeqTy);
  mlir::Value b = builder.create<fir::UndefOp>(loc, refSeqTy);
  mlir::Value result =
      builder.create<fir::UndefOp>(loc, fir::ReferenceType::get(eleTy));
  mlir::Value prod = fir::runtime::genDotProduct(builder, loc, a, b, result);
  if (fir::isa_complex(eleTy))
    checkCallOpFromResultBox(result, fctName, 3);
  else
    checkCallOp(prod.getDefiningOp(), fctName, 2);
}

TEST_F(RuntimeCallTest, genDotProduct) {
  testGenDotProduct(*firBuilder, f32Ty, "_FortranADotProductReal4");
  testGenDotProduct(*firBuilder, f64Ty, "_FortranADotProductReal8");
  testGenDotProduct(*firBuilder, f80Ty, "_FortranADotProductReal10");
  testGenDotProduct(*firBuilder, f128Ty, "_FortranADotProductReal16");
  testGenDotProduct(*firBuilder, i8Ty, "_FortranADotProductInteger1");
  testGenDotProduct(*firBuilder, i16Ty, "_FortranADotProductInteger2");
  testGenDotProduct(*firBuilder, i32Ty, "_FortranADotProductInteger4");
  testGenDotProduct(*firBuilder, i64Ty, "_FortranADotProductInteger8");
  testGenDotProduct(*firBuilder, i128Ty, "_FortranADotProductInteger16");
  testGenDotProduct(*firBuilder, c4Ty, "_FortranACppDotProductComplex4");
  testGenDotProduct(*firBuilder, c8Ty, "_FortranACppDotProductComplex8");
  testGenDotProduct(*firBuilder, c10Ty, "_FortranACppDotProductComplex10");
  testGenDotProduct(*firBuilder, c16Ty, "_FortranACppDotProductComplex16");
}

void checkGenMxxloc(fir::FirOpBuilder &builder, mlir::Type eleTy,
    void (*genFct)(fir::FirOpBuilder &, mlir::Location, mlir::Value,
        mlir::Value, mlir::Value, mlir::Value, mlir::Value),
    llvm::StringRef fctName, unsigned nbArgs) {
  mlir::Location loc = builder.getUnknownLoc();
  mlir::Type i32Ty = builder.getI32Type();
  mlir::Type seqTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, 10), eleTy);
  mlir::Type refSeqTy = fir::ReferenceType::get(seqTy);
  mlir::Value a = builder.create<fir::UndefOp>(loc, refSeqTy);
  mlir::Value result = builder.create<fir::UndefOp>(loc, seqTy);
  mlir::Value mask = builder.create<fir::UndefOp>(loc, seqTy);
  mlir::Value kind = builder.createIntegerConstant(loc, i32Ty, 1);
  mlir::Value back = builder.createIntegerConstant(loc, i32Ty, 1);
  genFct(builder, loc, result, a, mask, kind, back);
  checkCallOpFromResultBox(result, fctName, nbArgs);
}

TEST_F(RuntimeCallTest, genMaxlocTest) {
  checkGenMxxloc(*firBuilder, char1Ty, fir::runtime::genMaxloc,
      "_FortranAMaxlocCharacter", 5);
  checkGenMxxloc(*firBuilder, char2Ty, fir::runtime::genMaxloc,
      "_FortranAMaxlocCharacter", 5);
  checkGenMxxloc(*firBuilder, char4Ty, fir::runtime::genMaxloc,
      "_FortranAMaxlocCharacter", 5);
  checkGenMxxloc(
      *firBuilder, i8Ty, fir::runtime::genMaxloc, "_FortranAMaxlocInteger1", 5);
  checkGenMxxloc(*firBuilder, i16Ty, fir::runtime::genMaxloc,
      "_FortranAMaxlocInteger2", 5);
  checkGenMxxloc(*firBuilder, i32Ty, fir::runtime::genMaxloc,
      "_FortranAMaxlocInteger4", 5);
  checkGenMxxloc(*firBuilder, i64Ty, fir::runtime::genMaxloc,
      "_FortranAMaxlocInteger8", 5);
  checkGenMxxloc(*firBuilder, i128Ty, fir::runtime::genMaxloc,
      "_FortranAMaxlocInteger16", 5);
  checkGenMxxloc(
      *firBuilder, f32Ty, fir::runtime::genMaxloc, "_FortranAMaxlocReal4", 5);
  checkGenMxxloc(
      *firBuilder, f64Ty, fir::runtime::genMaxloc, "_FortranAMaxlocReal8", 5);
  checkGenMxxloc(
      *firBuilder, f80Ty, fir::runtime::genMaxloc, "_FortranAMaxlocReal10", 5);
  checkGenMxxloc(
      *firBuilder, f128Ty, fir::runtime::genMaxloc, "_FortranAMaxlocReal16", 5);
}

TEST_F(RuntimeCallTest, genMinlocTest) {
  checkGenMxxloc(*firBuilder, char1Ty, fir::runtime::genMinloc,
      "_FortranAMinlocCharacter", 5);
  checkGenMxxloc(*firBuilder, char2Ty, fir::runtime::genMinloc,
      "_FortranAMinlocCharacter", 5);
  checkGenMxxloc(*firBuilder, char4Ty, fir::runtime::genMinloc,
      "_FortranAMinlocCharacter", 5);
  checkGenMxxloc(
      *firBuilder, i8Ty, fir::runtime::genMinloc, "_FortranAMinlocInteger1", 5);
  checkGenMxxloc(*firBuilder, i16Ty, fir::runtime::genMinloc,
      "_FortranAMinlocInteger2", 5);
  checkGenMxxloc(*firBuilder, i32Ty, fir::runtime::genMinloc,
      "_FortranAMinlocInteger4", 5);
  checkGenMxxloc(*firBuilder, i64Ty, fir::runtime::genMinloc,
      "_FortranAMinlocInteger8", 5);
  checkGenMxxloc(*firBuilder, i128Ty, fir::runtime::genMinloc,
      "_FortranAMinlocInteger16", 5);
  checkGenMxxloc(
      *firBuilder, f32Ty, fir::runtime::genMinloc, "_FortranAMinlocReal4", 5);
  checkGenMxxloc(
      *firBuilder, f64Ty, fir::runtime::genMinloc, "_FortranAMinlocReal8", 5);
  checkGenMxxloc(
      *firBuilder, f80Ty, fir::runtime::genMinloc, "_FortranAMinlocReal10", 5);
  checkGenMxxloc(
      *firBuilder, f128Ty, fir::runtime::genMinloc, "_FortranAMinlocReal16", 5);
}

void checkGenMxxlocDim(fir::FirOpBuilder &builder,
    void (*genFct)(fir::FirOpBuilder &, mlir::Location, mlir::Value,
        mlir::Value, mlir::Value, mlir::Value, mlir::Value, mlir::Value),
    llvm::StringRef fctName, unsigned nbArgs) {
  mlir::Location loc = builder.getUnknownLoc();
  auto i32Ty = builder.getI32Type();
  mlir::Type seqTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, 10), i32Ty);
  mlir::Type refSeqTy = fir::ReferenceType::get(seqTy);
  mlir::Value a = builder.create<fir::UndefOp>(loc, refSeqTy);
  mlir::Value result = builder.create<fir::UndefOp>(loc, seqTy);
  mlir::Value mask = builder.create<fir::UndefOp>(loc, seqTy);
  mlir::Value kind = builder.createIntegerConstant(loc, i32Ty, 1);
  mlir::Value dim = builder.createIntegerConstant(loc, i32Ty, 1);
  mlir::Value back = builder.createIntegerConstant(loc, i32Ty, 1);
  genFct(builder, loc, result, a, dim, mask, kind, back);
  checkCallOpFromResultBox(result, fctName, nbArgs);
}

TEST_F(RuntimeCallTest, genMaxlocDimTest) {
  checkGenMxxlocDim(
      *firBuilder, fir::runtime::genMaxlocDim, "_FortranAMaxlocDim", 6);
}

TEST_F(RuntimeCallTest, genMinlocDimTest) {
  checkGenMxxlocDim(
      *firBuilder, fir::runtime::genMinlocDim, "_FortranAMinlocDim", 6);
}

void checkGenMxxvalChar(fir::FirOpBuilder &builder,
    void (*genFct)(fir::FirOpBuilder &, mlir::Location, mlir::Value,
        mlir::Value, mlir::Value),
    llvm::StringRef fctName, unsigned nbArgs) {
  mlir::Location loc = builder.getUnknownLoc();
  auto i32Ty = builder.getI32Type();
  mlir::Type seqTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, 10), i32Ty);
  mlir::Type refSeqTy = fir::ReferenceType::get(seqTy);
  mlir::Value a = builder.create<fir::UndefOp>(loc, refSeqTy);
  mlir::Value result = builder.create<fir::UndefOp>(loc, seqTy);
  mlir::Value mask = builder.create<fir::UndefOp>(loc, seqTy);
  genFct(builder, loc, result, a, mask);
  checkCallOpFromResultBox(result, fctName, nbArgs);
}

TEST_F(RuntimeCallTest, genMaxvalCharTest) {
  checkGenMxxvalChar(
      *firBuilder, fir::runtime::genMaxvalChar, "_FortranAMaxvalCharacter", 3);
}

TEST_F(RuntimeCallTest, genMinvalCharTest) {
  checkGenMxxvalChar(
      *firBuilder, fir::runtime::genMinvalChar, "_FortranAMinvalCharacter", 3);
}

void checkGen4argsDim(fir::FirOpBuilder &builder,
    void (*genFct)(fir::FirOpBuilder &, mlir::Location, mlir::Value,
        mlir::Value, mlir::Value, mlir::Value),
    llvm::StringRef fctName, unsigned nbArgs) {
  mlir::Location loc = builder.getUnknownLoc();
  auto i32Ty = builder.getI32Type();
  mlir::Type seqTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, 10), i32Ty);
  mlir::Type refSeqTy = fir::ReferenceType::get(seqTy);
  mlir::Value a = builder.create<fir::UndefOp>(loc, refSeqTy);
  mlir::Value result = builder.create<fir::UndefOp>(loc, seqTy);
  mlir::Value mask = builder.create<fir::UndefOp>(loc, seqTy);
  mlir::Value dim = builder.createIntegerConstant(loc, i32Ty, 1);
  genFct(builder, loc, result, a, dim, mask);
  checkCallOpFromResultBox(result, fctName, nbArgs);
}

TEST_F(RuntimeCallTest, genMaxvalDimTest) {
  checkGen4argsDim(
      *firBuilder, fir::runtime::genMaxvalDim, "_FortranAMaxvalDim", 4);
}

TEST_F(RuntimeCallTest, genMinvalDimTest) {
  checkGen4argsDim(
      *firBuilder, fir::runtime::genMinvalDim, "_FortranAMinvalDim", 4);
}

TEST_F(RuntimeCallTest, genProductDimTest) {
  checkGen4argsDim(
      *firBuilder, fir::runtime::genProductDim, "_FortranAProductDim", 4);
}

TEST_F(RuntimeCallTest, genSumDimTest) {
  checkGen4argsDim(*firBuilder, fir::runtime::genSumDim, "_FortranASumDim", 4);
}
