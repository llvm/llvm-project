//===- KindMappingTest.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Support/KindMapping.h"
#include "gtest/gtest.h"
#include <string>

using namespace fir;
namespace llvm {
struct fltSemantics;
} // namespace llvm

namespace mlir {
class MLIRContext;
} // namespace mlir

using Bitsize = fir::KindMapping::Bitsize;
using LLVMTypeID = fir::KindMapping::LLVMTypeID;

struct DefaultStringTests : public testing::Test {
public:
  void SetUp() { defaultString = new KindMapping(context); }
  void TearDown() { delete defaultString; }

  KindMapping *defaultString{};
  mlir::MLIRContext *context{};
};

struct CommandLineStringTests : public testing::Test {
public:
  void SetUp() {
    commandLineString = new KindMapping(context,
        "i10:80,l3:24,a1:8,r54:Double,c20:X86_FP80,r11:PPC_FP128,"
        "r12:FP128,r13:X86_FP80,r14:Double,r15:Float,r16:Half,r23:BFloat");
    clStringConflict =
        new KindMapping(context, "i10:80,i10:40,r54:Double,r54:X86_FP80");
  }
  void TearDown() {
    delete commandLineString;
    delete clStringConflict;
  }

  KindMapping *commandLineString{};
  KindMapping *clStringConflict{};
  mlir::MLIRContext *context{};
};

struct KindDefaultsTests : public testing::Test {
public:
  void SetUp() {
    defaultDefaultKinds = new KindMapping(context);
    overrideDefaultKinds =
        new KindMapping(context, {20, 121, 32, 133, 44, 145});
  }
  void TearDown() {
    delete defaultDefaultKinds;
    delete overrideDefaultKinds;
  }

  mlir::MLIRContext *context{};
  KindMapping *defaultDefaultKinds{};
  KindMapping *overrideDefaultKinds{};
};

TEST_F(DefaultStringTests, getIntegerBitsizeTest) {
  EXPECT_EQ(defaultString->getIntegerBitsize(10), 80u);
  EXPECT_EQ(defaultString->getIntegerBitsize(0), 0u);
}

TEST_F(DefaultStringTests, getCharacterBitsizeTest) {
  EXPECT_EQ(defaultString->getCharacterBitsize(10), 80u);
  EXPECT_EQ(defaultString->getCharacterBitsize(0), 0u);
}

TEST_F(DefaultStringTests, getLogicalBitsizeTest) {
  EXPECT_EQ(defaultString->getLogicalBitsize(10), 80u);
  // Unsigned values are expected
  std::string actual = std::to_string(defaultString->getLogicalBitsize(-10));
  std::string expect = "-80";
  EXPECT_NE(actual, expect);
}

TEST_F(DefaultStringTests, getRealTypeIDTest) {
  EXPECT_EQ(defaultString->getRealTypeID(2), LLVMTypeID::HalfTyID);
  EXPECT_EQ(defaultString->getRealTypeID(3), LLVMTypeID::BFloatTyID);
  EXPECT_EQ(defaultString->getRealTypeID(4), LLVMTypeID::FloatTyID);
  EXPECT_EQ(defaultString->getRealTypeID(8), LLVMTypeID::DoubleTyID);
  EXPECT_EQ(defaultString->getRealTypeID(10), LLVMTypeID::X86_FP80TyID);
  EXPECT_EQ(defaultString->getRealTypeID(16), LLVMTypeID::FP128TyID);
  // Default cases
  EXPECT_EQ(defaultString->getRealTypeID(-1), LLVMTypeID::FloatTyID);
  EXPECT_EQ(defaultString->getRealTypeID(1), LLVMTypeID::FloatTyID);
}

TEST_F(DefaultStringTests, getComplexTypeIDTest) {
  EXPECT_EQ(defaultString->getComplexTypeID(2), LLVMTypeID::HalfTyID);
  EXPECT_EQ(defaultString->getComplexTypeID(3), LLVMTypeID::BFloatTyID);
  EXPECT_EQ(defaultString->getComplexTypeID(4), LLVMTypeID::FloatTyID);
  EXPECT_EQ(defaultString->getComplexTypeID(8), LLVMTypeID::DoubleTyID);
  EXPECT_EQ(defaultString->getComplexTypeID(10), LLVMTypeID::X86_FP80TyID);
  EXPECT_EQ(defaultString->getComplexTypeID(16), LLVMTypeID::FP128TyID);
  // Default cases
  EXPECT_EQ(defaultString->getComplexTypeID(-1), LLVMTypeID::FloatTyID);
  EXPECT_EQ(defaultString->getComplexTypeID(1), LLVMTypeID::FloatTyID);
}

TEST_F(DefaultStringTests, getFloatSemanticsTest) {
  EXPECT_EQ(&defaultString->getFloatSemantics(2), &llvm::APFloat::IEEEhalf());
  EXPECT_EQ(&defaultString->getFloatSemantics(3), &llvm::APFloat::BFloat());
  EXPECT_EQ(&defaultString->getFloatSemantics(4), &llvm::APFloat::IEEEsingle());
  EXPECT_EQ(&defaultString->getFloatSemantics(8), &llvm::APFloat::IEEEdouble());
  EXPECT_EQ(&defaultString->getFloatSemantics(10),
      &llvm::APFloat::x87DoubleExtended());
  EXPECT_EQ(&defaultString->getFloatSemantics(16), &llvm::APFloat::IEEEquad());

  // Default cases
  EXPECT_EQ(
      &defaultString->getFloatSemantics(-1), &llvm::APFloat::IEEEsingle());
  EXPECT_EQ(&defaultString->getFloatSemantics(1), &llvm::APFloat::IEEEsingle());
}

TEST_F(CommandLineStringTests, getIntegerBitsizeTest) {
  // KEY is present in map.
  EXPECT_EQ(commandLineString->getIntegerBitsize(10), 80u);
  EXPECT_EQ(commandLineString->getCharacterBitsize(1), 8u);
  EXPECT_EQ(commandLineString->getLogicalBitsize(3), 24u);
  EXPECT_EQ(commandLineString->getComplexTypeID(20), LLVMTypeID::X86_FP80TyID);
  EXPECT_EQ(commandLineString->getRealTypeID(54), LLVMTypeID::DoubleTyID);
  EXPECT_EQ(commandLineString->getRealTypeID(11), LLVMTypeID::PPC_FP128TyID);
  EXPECT_EQ(&commandLineString->getFloatSemantics(11),
      &llvm::APFloat::PPCDoubleDouble());
  EXPECT_EQ(
      &commandLineString->getFloatSemantics(12), &llvm::APFloat::IEEEquad());
  EXPECT_EQ(&commandLineString->getFloatSemantics(13),
      &llvm::APFloat::x87DoubleExtended());
  EXPECT_EQ(
      &commandLineString->getFloatSemantics(14), &llvm::APFloat::IEEEdouble());
  EXPECT_EQ(
      &commandLineString->getFloatSemantics(15), &llvm::APFloat::IEEEsingle());
  EXPECT_EQ(
      &commandLineString->getFloatSemantics(16), &llvm::APFloat::IEEEhalf());
  EXPECT_EQ(
      &commandLineString->getFloatSemantics(23), &llvm::APFloat::BFloat());

  // Converts to default case
  EXPECT_EQ(
      &commandLineString->getFloatSemantics(20), &llvm::APFloat::IEEEsingle());

  // KEY is absent from map, Default values are expected.
  EXPECT_EQ(commandLineString->getIntegerBitsize(9), 72u);
  EXPECT_EQ(commandLineString->getCharacterBitsize(9), 72u);
  EXPECT_EQ(commandLineString->getLogicalBitsize(9), 72u);
  EXPECT_EQ(commandLineString->getComplexTypeID(9), LLVMTypeID::FloatTyID);
  EXPECT_EQ(commandLineString->getRealTypeID(9), LLVMTypeID::FloatTyID);

  // KEY repeats in map.
  EXPECT_NE(clStringConflict->getIntegerBitsize(10), 80u);
  EXPECT_NE(clStringConflict->getRealTypeID(10), LLVMTypeID::DoubleTyID);
}

TEST(KindMappingDeathTests, mapTest) {
  mlir::MLIRContext *context{};
  // Catch parsing errors
  ASSERT_DEATH(new KindMapping(context, "r10:Double,r20:Doubl"), "");
  ASSERT_DEATH(new KindMapping(context, "10:Double"), "");
  ASSERT_DEATH(new KindMapping(context, "rr:Double"), "");
  ASSERT_DEATH(new KindMapping(context, "rr:"), "");
  ASSERT_DEATH(new KindMapping(context, "rr:Double MoreContent"), "");
  // length of 'size' > 10
  ASSERT_DEATH(new KindMapping(context, "i11111111111:10"), "");
}

TEST_F(KindDefaultsTests, getIntegerBitsizeTest) {
  EXPECT_EQ(defaultDefaultKinds->defaultCharacterKind(), 1u);
  EXPECT_EQ(defaultDefaultKinds->defaultComplexKind(), 4u);
  EXPECT_EQ(defaultDefaultKinds->defaultDoubleKind(), 8u);
  EXPECT_EQ(defaultDefaultKinds->defaultIntegerKind(), 4u);
  EXPECT_EQ(defaultDefaultKinds->defaultLogicalKind(), 4u);
  EXPECT_EQ(defaultDefaultKinds->defaultRealKind(), 4u);

  EXPECT_EQ(overrideDefaultKinds->defaultCharacterKind(), 20u);
  EXPECT_EQ(overrideDefaultKinds->defaultComplexKind(), 121u);
  EXPECT_EQ(overrideDefaultKinds->defaultDoubleKind(), 32u);
  EXPECT_EQ(overrideDefaultKinds->defaultIntegerKind(), 133u);
  EXPECT_EQ(overrideDefaultKinds->defaultLogicalKind(), 44u);
  EXPECT_EQ(overrideDefaultKinds->defaultRealKind(), 145u);
}

// main() from gtest_main
