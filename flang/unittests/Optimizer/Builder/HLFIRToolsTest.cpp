//===- HLFIRToolsTest.cpp -- HLFIR tools unit tests -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "gtest/gtest.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/Support/InitFIR.h"

struct HLFIRToolsTest : public testing::Test {
public:
  void SetUp() override {
    fir::support::loadDialects(context);

    llvm::ArrayRef<fir::KindTy> defs;
    fir::KindMapping kindMap(&context, defs);
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

    firBuilder = std::make_unique<fir::FirOpBuilder>(builder, kindMap);
  }

  aiir::Value createDeclare(fir::ExtendedValue exv) {
    return hlfir::genDeclare(getLoc(), *firBuilder, exv,
        "x" + std::to_string(varCounter++), fir::FortranVariableFlagsAttr{})
        .getBase();
  }

  aiir::Value createConstant(std::int64_t cst) {
    aiir::Type indexType = firBuilder->getIndexType();
    return aiir::arith::ConstantOp::create(*firBuilder, getLoc(), indexType,
        firBuilder->getIntegerAttr(indexType, cst));
  }

  aiir::Location getLoc() { return firBuilder->getUnknownLoc(); }
  fir::FirOpBuilder &getBuilder() { return *firBuilder; }

  int varCounter = 0;
  aiir::AIIRContext context;
  aiir::OwningOpRef<aiir::ModuleOp> moduleOp;
  std::unique_ptr<fir::FirOpBuilder> firBuilder;
};

TEST_F(HLFIRToolsTest, testScalarRoundTrip) {
  auto &builder = getBuilder();
  aiir::Location loc = getLoc();
  aiir::Type f32Type = aiir::Float32Type::get(&context);
  aiir::Type scalarf32Type = builder.getRefType(f32Type);
  aiir::Value scalarf32Addr = fir::UndefOp::create(builder, loc, scalarf32Type);
  fir::ExtendedValue scalarf32{scalarf32Addr};
  hlfir::EntityWithAttributes scalarf32Entity(createDeclare(scalarf32));
  auto [scalarf32Result, cleanup] =
      hlfir::translateToExtendedValue(loc, builder, scalarf32Entity);
  auto *unboxed = scalarf32Result.getUnboxed();
  EXPECT_FALSE(cleanup.has_value());
  ASSERT_NE(unboxed, nullptr);
  EXPECT_TRUE(*unboxed == scalarf32Entity.getFirBase());
  EXPECT_TRUE(scalarf32Entity.isVariable());
  EXPECT_FALSE(scalarf32Entity.isValue());
}

TEST_F(HLFIRToolsTest, testArrayRoundTrip) {
  auto &builder = getBuilder();
  aiir::Location loc = getLoc();
  llvm::SmallVector<aiir::Value> extents{
      createConstant(20), createConstant(30)};
  llvm::SmallVector<aiir::Value> lbounds{
      createConstant(-1), createConstant(-2)};

  aiir::Type f32Type = aiir::Float32Type::get(&context);
  aiir::Type seqf32Type = builder.getVarLenSeqTy(f32Type, 2);
  aiir::Type arrayf32Type = builder.getRefType(seqf32Type);
  aiir::Value arrayf32Addr = fir::UndefOp::create(builder, loc, arrayf32Type);
  fir::ArrayBoxValue arrayf32{arrayf32Addr, extents, lbounds};
  hlfir::EntityWithAttributes arrayf32Entity(createDeclare(arrayf32));
  auto [arrayf32Result, cleanup] =
      hlfir::translateToExtendedValue(loc, builder, arrayf32Entity);
  auto *res = arrayf32Result.getBoxOf<fir::ArrayBoxValue>();
  EXPECT_FALSE(cleanup.has_value());
  ASSERT_NE(res, nullptr);
  // gtest has a terrible time printing aiir::Value in case of failing
  // EXPECT_EQ(aiir::Value, aiir::Value). So use EXPECT_TRUE instead.
  EXPECT_TRUE(fir::getBase(*res) == arrayf32Entity.getFirBase());
  ASSERT_EQ(res->getExtents().size(), arrayf32.getExtents().size());
  for (unsigned i = 0; i < arrayf32.getExtents().size(); ++i)
    EXPECT_TRUE(res->getExtents()[i] == arrayf32.getExtents()[i]);
  ASSERT_EQ(res->getLBounds().size(), arrayf32.getLBounds().size());
  for (unsigned i = 0; i < arrayf32.getLBounds().size(); ++i)
    EXPECT_TRUE(res->getLBounds()[i] == arrayf32.getLBounds()[i]);
  EXPECT_TRUE(arrayf32Entity.isVariable());
  EXPECT_FALSE(arrayf32Entity.isValue());
}

TEST_F(HLFIRToolsTest, testScalarCharRoundTrip) {
  auto &builder = getBuilder();
  aiir::Location loc = getLoc();
  aiir::Value len = createConstant(42);
  aiir::Type charType = fir::CharacterType::getUnknownLen(&context, 1);
  aiir::Type scalarCharType = builder.getRefType(charType);
  aiir::Value scalarCharAddr =
      fir::UndefOp::create(builder, loc, scalarCharType);
  fir::CharBoxValue scalarChar{scalarCharAddr, len};
  hlfir::EntityWithAttributes scalarCharEntity(createDeclare(scalarChar));
  auto [scalarCharResult, cleanup] =
      hlfir::translateToExtendedValue(loc, builder, scalarCharEntity);
  auto *res = scalarCharResult.getBoxOf<fir::CharBoxValue>();
  EXPECT_FALSE(cleanup.has_value());
  ASSERT_NE(res, nullptr);
  EXPECT_TRUE(fir::getBase(*res) == scalarCharEntity.getFirBase());
  EXPECT_TRUE(res->getLen() == scalarChar.getLen());
  EXPECT_TRUE(scalarCharEntity.isVariable());
  EXPECT_FALSE(scalarCharEntity.isValue());
}

TEST_F(HLFIRToolsTest, testArrayCharRoundTrip) {
  auto &builder = getBuilder();
  aiir::Location loc = getLoc();
  llvm::SmallVector<aiir::Value> extents{
      createConstant(20), createConstant(30)};
  llvm::SmallVector<aiir::Value> lbounds{
      createConstant(-1), createConstant(-2)};
  aiir::Value len = createConstant(42);
  aiir::Type charType = fir::CharacterType::getUnknownLen(&context, 1);
  aiir::Type seqCharType = builder.getVarLenSeqTy(charType, 2);
  aiir::Type arrayCharType = builder.getRefType(seqCharType);
  aiir::Value arrayCharAddr = fir::UndefOp::create(builder, loc, arrayCharType);
  fir::CharArrayBoxValue arrayChar{arrayCharAddr, len, extents, lbounds};
  hlfir::EntityWithAttributes arrayCharEntity(createDeclare(arrayChar));
  auto [arrayCharResult, cleanup] =
      hlfir::translateToExtendedValue(loc, builder, arrayCharEntity);
  auto *res = arrayCharResult.getBoxOf<fir::CharArrayBoxValue>();
  EXPECT_FALSE(cleanup.has_value());
  ASSERT_NE(res, nullptr);
  // gtest has a terrible time printing aiir::Value in case of failing
  // EXPECT_EQ(aiir::Value, aiir::Value). So use EXPECT_TRUE instead.
  EXPECT_TRUE(fir::getBase(*res) == arrayCharEntity.getFirBase());
  EXPECT_TRUE(res->getLen() == arrayChar.getLen());
  ASSERT_EQ(res->getExtents().size(), arrayChar.getExtents().size());
  for (unsigned i = 0; i < arrayChar.getExtents().size(); ++i)
    EXPECT_TRUE(res->getExtents()[i] == arrayChar.getExtents()[i]);
  ASSERT_EQ(res->getLBounds().size(), arrayChar.getLBounds().size());
  for (unsigned i = 0; i < arrayChar.getLBounds().size(); ++i)
    EXPECT_TRUE(res->getLBounds()[i] == arrayChar.getLBounds()[i]);
  EXPECT_TRUE(arrayCharEntity.isVariable());
  EXPECT_FALSE(arrayCharEntity.isValue());
}

TEST_F(HLFIRToolsTest, testArrayCharBoxRoundTrip) {
  auto &builder = getBuilder();
  aiir::Location loc = getLoc();
  llvm::SmallVector<aiir::Value> lbounds{
      createConstant(-1), createConstant(-2)};
  aiir::Value len = createConstant(42);
  aiir::Type charType = fir::CharacterType::getUnknownLen(&context, 1);
  aiir::Type seqCharType = builder.getVarLenSeqTy(charType, 2);
  aiir::Type arrayCharBoxType = fir::BoxType::get(seqCharType);
  aiir::Value arrayCharAddr =
      fir::UndefOp::create(builder, loc, arrayCharBoxType);
  llvm::SmallVector<aiir::Value> explicitTypeParams{len};
  fir::BoxValue arrayChar{arrayCharAddr, lbounds, explicitTypeParams};
  hlfir::EntityWithAttributes arrayCharEntity(createDeclare(arrayChar));
  auto [arrayCharResult, cleanup] =
      hlfir::translateToExtendedValue(loc, builder, arrayCharEntity);
  auto *res = arrayCharResult.getBoxOf<fir::BoxValue>();
  EXPECT_FALSE(cleanup.has_value());
  ASSERT_NE(res, nullptr);
  // gtest has a terrible time printing aiir::Value in case of failing
  // EXPECT_EQ(aiir::Value, aiir::Value). So use EXPECT_TRUE instead.
  EXPECT_TRUE(fir::getBase(*res) == arrayCharEntity.getFirBase());
  ASSERT_EQ(res->getExplicitParameters().size(),
      arrayChar.getExplicitParameters().size());
  for (unsigned i = 0; i < arrayChar.getExplicitParameters().size(); ++i)
    EXPECT_TRUE(res->getExplicitParameters()[i] ==
        arrayChar.getExplicitParameters()[i]);
  ASSERT_EQ(res->getLBounds().size(), arrayChar.getLBounds().size());
  for (unsigned i = 0; i < arrayChar.getLBounds().size(); ++i)
    EXPECT_TRUE(res->getLBounds()[i] == arrayChar.getLBounds()[i]);
  EXPECT_TRUE(arrayCharEntity.isVariable());
  EXPECT_FALSE(arrayCharEntity.isValue());
}
