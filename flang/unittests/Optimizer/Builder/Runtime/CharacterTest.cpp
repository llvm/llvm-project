//===- CharacterTest.cpp -- Character runtime builder unit tests ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/Character.h"
#include "RuntimeCallTestBase.h"
#include "gtest/gtest.h"
#include "flang/Optimizer/Builder/Character.h"

using namespace mlir;

TEST_F(RuntimeCallTest, genAdjustLTest) {
  auto loc = firBuilder->getUnknownLoc();
  mlir::Value result = fir::UndefOp::create(*firBuilder, loc, boxTy);
  mlir::Value string = fir::UndefOp::create(*firBuilder, loc, boxTy);
  fir::runtime::genAdjustL(*firBuilder, loc, result, string);
  checkCallOpFromResultBox(result, "_FortranAAdjustl", 2);
}

TEST_F(RuntimeCallTest, genAdjustRTest) {
  auto loc = firBuilder->getUnknownLoc();
  mlir::Value result = fir::UndefOp::create(*firBuilder, loc, boxTy);
  mlir::Value string = fir::UndefOp::create(*firBuilder, loc, boxTy);
  fir::runtime::genAdjustR(*firBuilder, loc, result, string);
  checkCallOpFromResultBox(result, "_FortranAAdjustr", 2);
}

void checkCharCompare1(
    fir::FirOpBuilder &builder, mlir::Type type, llvm::StringRef fctName) {
  auto loc = builder.getUnknownLoc();
  mlir::Type i32Ty = IntegerType::get(builder.getContext(), 32);
  mlir::Value lhsBuff = fir::UndefOp::create(builder, loc, type);
  mlir::Value lhsLen = fir::UndefOp::create(builder, loc, i32Ty);
  mlir::Value rhsBuff = fir::UndefOp::create(builder, loc, type);
  mlir::Value rhsLen = fir::UndefOp::create(builder, loc, i32Ty);
  mlir::Value res = fir::runtime::genCharCompare(builder, loc,
      mlir::arith::CmpIPredicate::eq, lhsBuff, lhsLen, rhsBuff, rhsLen);
  checkCallOpFromResultBox(lhsBuff, fctName, 4, /*addLocArgs=*/false);
  EXPECT_TRUE(mlir::isa<mlir::arith::CmpIOp>(res.getDefiningOp()));
}

void checkCharCompare1AllTypeForKind(
    fir::FirOpBuilder &builder, llvm::StringRef fctName, unsigned kind) {
  mlir::Type charTy = fir::CharacterType::get(builder.getContext(), kind, 10);
  mlir::Type seqCharTy = fir::SequenceType::get(charTy, 10);
  mlir::Type refCharTy = fir::ReferenceType::get(charTy);
  mlir::Type boxCharTy = fir::BoxCharType::get(builder.getContext(), kind);
  mlir::Type boxTy = fir::BoxType::get(charTy);
  checkCharCompare1(builder, charTy, fctName);
  checkCharCompare1(builder, seqCharTy, fctName);
  checkCharCompare1(builder, refCharTy, fctName);
  checkCharCompare1(builder, boxCharTy, fctName);
  checkCharCompare1(builder, boxTy, fctName);
}

TEST_F(RuntimeCallTest, genCharCompar1Test) {
  checkCharCompare1AllTypeForKind(
      *firBuilder, "_FortranACharacterCompareScalar1", 1);
  checkCharCompare1AllTypeForKind(
      *firBuilder, "_FortranACharacterCompareScalar2", 2);
  checkCharCompare1AllTypeForKind(
      *firBuilder, "_FortranACharacterCompareScalar4", 4);
}

void checkCharCompare2(
    fir::FirOpBuilder &builder, llvm::StringRef fctName, unsigned kind) {
  auto loc = builder.getUnknownLoc();
  fir::factory::CharacterExprHelper charHelper(builder, loc);
  mlir::Type i32Ty = IntegerType::get(builder.getContext(), 32);
  mlir::Type boxCharTy = fir::BoxCharType::get(builder.getContext(), kind);
  mlir::Value lhsBuff = fir::UndefOp::create(builder, loc, boxCharTy);
  mlir::Value lhsLen = fir::UndefOp::create(builder, loc, i32Ty);
  mlir::Value rhsBuff = fir::UndefOp::create(builder, loc, boxCharTy);
  mlir::Value rhsLen = fir::UndefOp::create(builder, loc, i32Ty);
  fir::ExtendedValue lhs = charHelper.toExtendedValue(lhsBuff, lhsLen);
  fir::ExtendedValue rhs = charHelper.toExtendedValue(rhsBuff, rhsLen);
  mlir::Value res = fir::runtime::genCharCompare(
      builder, loc, mlir::arith::CmpIPredicate::eq, lhs, rhs);
  EXPECT_TRUE(mlir::isa<mlir::arith::CmpIOp>(res.getDefiningOp()));
  auto cmpOp = mlir::dyn_cast<mlir::arith::CmpIOp>(res.getDefiningOp());
  checkCallOp(cmpOp.getLhs().getDefiningOp(), fctName, 4, /*addLocArgs=*/false);
  auto allocas = res.getParentBlock()->getOps<fir::AllocaOp>();
  EXPECT_TRUE(allocas.empty());
}

TEST_F(RuntimeCallTest, genCharCompare2Test) {
  checkCharCompare2(*firBuilder, "_FortranACharacterCompareScalar1", 1);
  checkCharCompare2(*firBuilder, "_FortranACharacterCompareScalar2", 2);
  checkCharCompare2(*firBuilder, "_FortranACharacterCompareScalar4", 4);
}

void checkGenIndex(
    fir::FirOpBuilder &builder, llvm::StringRef fctName, unsigned kind) {
  auto loc = builder.getUnknownLoc();
  mlir::Type i32Ty = IntegerType::get(builder.getContext(), 32);
  mlir::Value stringBase = fir::UndefOp::create(builder, loc, i32Ty);
  mlir::Value stringLen = fir::UndefOp::create(builder, loc, i32Ty);
  mlir::Value substringBase = fir::UndefOp::create(builder, loc, i32Ty);
  mlir::Value substringLen = fir::UndefOp::create(builder, loc, i32Ty);
  mlir::Value back = fir::UndefOp::create(builder, loc, i32Ty);
  mlir::Value res = fir::runtime::genIndex(builder, loc, kind, stringBase,
      stringLen, substringBase, substringLen, back);
  checkCallOp(res.getDefiningOp(), fctName, 5, /*addLocArgs=*/false);
}

TEST_F(RuntimeCallTest, genIndexTest) {
  checkGenIndex(*firBuilder, "_FortranAIndex1", 1);
  checkGenIndex(*firBuilder, "_FortranAIndex2", 2);
  checkGenIndex(*firBuilder, "_FortranAIndex4", 4);
}

TEST_F(RuntimeCallTest, genIndexDescriptorTest) {
  auto loc = firBuilder->getUnknownLoc();
  mlir::Value resultBox = fir::UndefOp::create(*firBuilder, loc, boxTy);
  mlir::Value stringBox = fir::UndefOp::create(*firBuilder, loc, boxTy);
  mlir::Value substringBox = fir::UndefOp::create(*firBuilder, loc, boxTy);
  mlir::Value backOpt = fir::UndefOp::create(*firBuilder, loc, boxTy);
  mlir::Value kind = fir::UndefOp::create(*firBuilder, loc, i32Ty);
  fir::runtime::genIndexDescriptor(
      *firBuilder, loc, resultBox, stringBox, substringBox, backOpt, kind);
  checkCallOpFromResultBox(resultBox, "_FortranAIndex", 5);
}

TEST_F(RuntimeCallTest, genRepeatTest) {
  auto loc = firBuilder->getUnknownLoc();
  mlir::Value resultBox = fir::UndefOp::create(*firBuilder, loc, boxTy);
  mlir::Value stringBox = fir::UndefOp::create(*firBuilder, loc, boxTy);
  mlir::Value ncopies = fir::UndefOp::create(*firBuilder, loc, i32Ty);
  fir::runtime::genRepeat(*firBuilder, loc, resultBox, stringBox, ncopies);
  checkCallOpFromResultBox(resultBox, "_FortranARepeat", 3);
}

TEST_F(RuntimeCallTest, genTrimTest) {
  auto loc = firBuilder->getUnknownLoc();
  mlir::Value resultBox = fir::UndefOp::create(*firBuilder, loc, boxTy);
  mlir::Value stringBox = fir::UndefOp::create(*firBuilder, loc, boxTy);
  fir::runtime::genTrim(*firBuilder, loc, resultBox, stringBox);
  checkCallOpFromResultBox(resultBox, "_FortranATrim", 2);
}

TEST_F(RuntimeCallTest, genScanDescriptorTest) {
  auto loc = firBuilder->getUnknownLoc();
  mlir::Value resultBox = fir::UndefOp::create(*firBuilder, loc, boxTy);
  mlir::Value stringBox = fir::UndefOp::create(*firBuilder, loc, boxTy);
  mlir::Value setBox = fir::UndefOp::create(*firBuilder, loc, boxTy);
  mlir::Value backBox = fir::UndefOp::create(*firBuilder, loc, boxTy);
  mlir::Value kind = fir::UndefOp::create(*firBuilder, loc, i32Ty);
  fir::runtime::genScanDescriptor(
      *firBuilder, loc, resultBox, stringBox, setBox, backBox, kind);
  checkCallOpFromResultBox(resultBox, "_FortranAScan", 5);
}

void checkGenScan(
    fir::FirOpBuilder &builder, llvm::StringRef fctName, unsigned kind) {
  auto loc = builder.getUnknownLoc();
  mlir::Type charTy = fir::CharacterType::get(builder.getContext(), kind, 10);
  mlir::Type boxTy = fir::BoxType::get(charTy);
  mlir::Type i32Ty = IntegerType::get(builder.getContext(), 32);
  mlir::Value stringBase = fir::UndefOp::create(builder, loc, boxTy);
  mlir::Value stringLen = fir::UndefOp::create(builder, loc, i32Ty);
  mlir::Value setBase = fir::UndefOp::create(builder, loc, boxTy);
  mlir::Value setLen = fir::UndefOp::create(builder, loc, i32Ty);
  mlir::Value back = fir::UndefOp::create(builder, loc, i32Ty);
  mlir::Value res = fir::runtime::genScan(
      builder, loc, kind, stringBase, stringLen, setBase, setLen, back);
  checkCallOp(res.getDefiningOp(), fctName, 5, /*addLocArgs=*/false);
}

TEST_F(RuntimeCallTest, genScanTest) {
  checkGenScan(*firBuilder, "_FortranAScan1", 1);
  checkGenScan(*firBuilder, "_FortranAScan2", 2);
  checkGenScan(*firBuilder, "_FortranAScan4", 4);
}

TEST_F(RuntimeCallTest, genVerifyDescriptorTest) {
  auto loc = firBuilder->getUnknownLoc();
  mlir::Value resultBox = fir::UndefOp::create(*firBuilder, loc, boxTy);
  mlir::Value stringBox = fir::UndefOp::create(*firBuilder, loc, boxTy);
  mlir::Value setBox = fir::UndefOp::create(*firBuilder, loc, boxTy);
  mlir::Value backBox = fir::UndefOp::create(*firBuilder, loc, boxTy);
  mlir::Value kind = fir::UndefOp::create(*firBuilder, loc, i32Ty);
  fir::runtime::genVerifyDescriptor(
      *firBuilder, loc, resultBox, stringBox, setBox, backBox, kind);
  checkCallOpFromResultBox(resultBox, "_FortranAVerify", 5);
}

void checkGenVerify(
    fir::FirOpBuilder &builder, llvm::StringRef fctName, unsigned kind) {
  auto loc = builder.getUnknownLoc();
  mlir::Type charTy = fir::CharacterType::get(builder.getContext(), kind, 10);
  mlir::Type boxTy = fir::BoxType::get(charTy);
  mlir::Type i32Ty = IntegerType::get(builder.getContext(), 32);
  mlir::Value stringBase = fir::UndefOp::create(builder, loc, boxTy);
  mlir::Value stringLen = fir::UndefOp::create(builder, loc, i32Ty);
  mlir::Value setBase = fir::UndefOp::create(builder, loc, boxTy);
  mlir::Value setLen = fir::UndefOp::create(builder, loc, i32Ty);
  mlir::Value back = fir::UndefOp::create(builder, loc, i32Ty);
  mlir::Value res = fir::runtime::genVerify(
      builder, loc, kind, stringBase, stringLen, setBase, setLen, back);
  checkCallOp(res.getDefiningOp(), fctName, 5, /*addLocArgs=*/false);
}

TEST_F(RuntimeCallTest, genVerifyTest) {
  checkGenVerify(*firBuilder, "_FortranAVerify1", 1);
  checkGenVerify(*firBuilder, "_FortranAVerify2", 2);
  checkGenVerify(*firBuilder, "_FortranAVerify4", 4);
}
