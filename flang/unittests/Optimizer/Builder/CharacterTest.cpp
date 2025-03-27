//===- CharacterTest.cpp -- CharacterExprHelper unit tests ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Character.h"
#include "gtest/gtest.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/Support/InitFIR.h"

struct CharacterTest : public testing::Test {
public:
  void SetUp() override {
    fir::support::loadDialects(context);

    kindMap = std::make_unique<fir::KindMapping>(&context,
        "i10:80,l3:24,a1:8,r54:Double,c20:X86_FP80,r11:PPC_FP128,"
        "r12:FP128,r13:X86_FP80,r14:Double,r15:Float,r16:Half,r23:BFloat");
    mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();

    // Set up a Module with a dummy function operation inside.
    // Set the insertion point in the function entry block.
    moduleOp = builder.create<mlir::ModuleOp>(loc);
    builder.setInsertionPointToStart(moduleOp->getBody());
    mlir::func::FuncOp func = builder.create<mlir::func::FuncOp>(
        loc, "func1", builder.getFunctionType(std::nullopt, std::nullopt));
    auto *entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    firBuilder = std::make_unique<fir::FirOpBuilder>(builder, *kindMap);
  }

  fir::FirOpBuilder &getBuilder() { return *firBuilder; }

  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> moduleOp;
  std::unique_ptr<fir::KindMapping> kindMap;
  std::unique_ptr<fir::FirOpBuilder> firBuilder;
};

TEST_F(CharacterTest, smallUtilityFunctions) {
  auto builder = getBuilder();
  auto loc = builder.getUnknownLoc();
  llvm::StringRef strValue("onestringliteral");
  auto strLit = fir::factory::createStringLiteral(builder, loc, strValue);
  EXPECT_TRUE(
      fir::factory::CharacterExprHelper::hasConstantLengthInType(strLit));
  auto ty = strLit.getCharBox()->getAddr().getType();
  EXPECT_TRUE(fir::factory::CharacterExprHelper::isCharacterScalar(ty));
  EXPECT_EQ(fir::factory::CharacterExprHelper::getCharacterOrSequenceKind(ty),
      fir::factory::CharacterExprHelper::getCharacterKind(ty));
}

TEST_F(CharacterTest, createSubstring) {
  auto builder = getBuilder();
  auto loc = builder.getUnknownLoc();
  auto charHelper = fir::factory::CharacterExprHelper(builder, loc);
  llvm::StringRef data("a dummy string to test substring");
  auto str = fir::factory::createStringLiteral(builder, loc, data);
  auto lb = builder.createIntegerConstant(loc, builder.getI64Type(), 18);
  auto ub = builder.createIntegerConstant(loc, builder.getI64Type(), 22);
  auto substr = charHelper.createSubstring(*str.getCharBox(), {lb, ub});
  EXPECT_FALSE(
      fir::factory::CharacterExprHelper::hasConstantLengthInType(substr));
  EXPECT_FALSE(charHelper.getCharacterType(substr).hasConstantLen());
  EXPECT_FALSE(fir::factory::CharacterExprHelper::isArray(
      charHelper.getCharacterType(substr)));
}
