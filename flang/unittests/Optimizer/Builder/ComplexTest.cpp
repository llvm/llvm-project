//===- ComplexExprTest.cpp -- ComplexExpr unit tests ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Complex.h"
#include "gtest/gtest.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/Support/InitFIR.h"

struct ComplexTest : public testing::Test {
public:
  void SetUp() override {
    fir::support::loadDialects(context);

    aiir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();

    // Set up a Module with a dummy function operation inside.
    // Set the insertion point in the function entry block.
    moduleOp = aiir::ModuleOp::create(builder, loc);
    builder.setInsertionPointToStart(moduleOp->getBody());
    aiir::func::FuncOp func = aiir::func::FuncOp::create(
        builder, loc, "func1", builder.getFunctionType({}, {}));
    auto *entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    kindMap = std::make_unique<fir::KindMapping>(&context);
    firBuilder = std::make_unique<fir::FirOpBuilder>(builder, *kindMap);
    helper = std::make_unique<fir::factory::Complex>(*firBuilder, loc);

    // Init commonly used types
    realTy1 = aiir::Float32Type::get(&context);
    complexTy1 = aiir::ComplexType::get(realTy1);
    integerTy1 = aiir::IntegerType::get(&context, 32);

    // Create commonly used reals
    rOne = firBuilder->createRealConstant(loc, realTy1, 1u);
    rTwo = firBuilder->createRealConstant(loc, realTy1, 2u);
    rThree = firBuilder->createRealConstant(loc, realTy1, 3u);
    rFour = firBuilder->createRealConstant(loc, realTy1, 4u);
  }

  aiir::AIIRContext context;
  aiir::OwningOpRef<aiir::ModuleOp> moduleOp;
  std::unique_ptr<fir::KindMapping> kindMap;
  std::unique_ptr<fir::FirOpBuilder> firBuilder;
  std::unique_ptr<fir::factory::Complex> helper;

  // Commonly used real/complex/integer types
  aiir::FloatType realTy1;
  aiir::ComplexType complexTy1;
  aiir::IntegerType integerTy1;

  // Commonly used real numbers
  aiir::Value rOne;
  aiir::Value rTwo;
  aiir::Value rThree;
  aiir::Value rFour;
};

TEST_F(ComplexTest, verifyTypes) {
  aiir::Value cVal1 = helper->createComplex(complexTy1, rOne, rTwo);
  EXPECT_TRUE(fir::isa_complex(cVal1.getType()));
  EXPECT_TRUE(fir::isa_real(helper->getComplexPartType(cVal1)));

  aiir::Value real1 = helper->extractComplexPart(cVal1, /*isImagPart=*/false);
  aiir::Value imag1 = helper->extractComplexPart(cVal1, /*isImagPart=*/true);
  EXPECT_EQ(realTy1, real1.getType());
  EXPECT_EQ(realTy1, imag1.getType());

  aiir::Value cVal3 =
      helper->insertComplexPart(cVal1, rThree, /*isImagPart=*/false);
  aiir::Value cVal4 =
      helper->insertComplexPart(cVal3, rFour, /*isImagPart=*/true);
  EXPECT_TRUE(fir::isa_complex(cVal4.getType()));
  EXPECT_TRUE(fir::isa_real(helper->getComplexPartType(cVal4)));
}

TEST_F(ComplexTest, verifyConvertWithSemantics) {
  auto loc = firBuilder->getUnknownLoc();
  rOne = firBuilder->createRealConstant(loc, realTy1, 1u);
  // Convert real to complex
  aiir::Value v1 = firBuilder->convertWithSemantics(loc, complexTy1, rOne);
  EXPECT_TRUE(fir::isa_complex(v1.getType()));

  // Convert complex to integer
  aiir::Value v2 = firBuilder->convertWithSemantics(loc, integerTy1, v1);
  EXPECT_TRUE(aiir::isa<aiir::IntegerType>(v2.getType()));
  EXPECT_TRUE(aiir::dyn_cast<fir::ConvertOp>(v2.getDefiningOp()));
}
