//===- FIRBuilderTest.cpp -- FIRBuilder unit tests ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "gtest/gtest.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Support/InitFIR.h"
#include "flang/Optimizer/Support/KindMapping.h"

struct FIRBuilderTest : public testing::Test {
public:
  void SetUp() override {
    fir::KindMapping kindMap(&context);
    mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();

    // Set up a Module with a dummy function operation inside.
    // Set the insertion point in the function entry block.
    mlir::ModuleOp mod = builder.create<mlir::ModuleOp>(loc);
    mlir::FuncOp func = mlir::FuncOp::create(
        loc, "func1", builder.getFunctionType(llvm::None, llvm::None));
    auto *entryBlock = func.addEntryBlock();
    mod.push_back(mod);
    builder.setInsertionPointToStart(entryBlock);

    fir::support::loadDialects(context);
    firBuilder = std::make_unique<fir::FirOpBuilder>(mod, kindMap);
  }

  fir::FirOpBuilder &getBuilder() { return *firBuilder; }

  mlir::MLIRContext context;
  std::unique_ptr<fir::FirOpBuilder> firBuilder;
};

static arith::CmpIOp createCondition(fir::FirOpBuilder &builder) {
  auto loc = builder.getUnknownLoc();
  auto zero1 = builder.createIntegerConstant(loc, builder.getIndexType(), 0);
  auto zero2 = builder.createIntegerConstant(loc, builder.getIndexType(), 0);
  return builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, zero1, zero2);
}

static void checkIntegerConstant(mlir::Value value, mlir::Type ty, int64_t v) {
  EXPECT_TRUE(mlir::isa<ConstantOp>(value.getDefiningOp()));
  auto cstOp = dyn_cast<ConstantOp>(value.getDefiningOp());
  EXPECT_EQ(ty, cstOp.getType());
  auto valueAttr = cstOp.getValue().dyn_cast_or_null<IntegerAttr>();
  EXPECT_EQ(v, valueAttr.getInt());
}

//===----------------------------------------------------------------------===//
// IfBuilder tests
//===----------------------------------------------------------------------===//

TEST_F(FIRBuilderTest, genIfThen) {
  auto builder = getBuilder();
  auto loc = builder.getUnknownLoc();
  auto cdt = createCondition(builder);
  auto ifBuilder = builder.genIfThen(loc, cdt);
  EXPECT_FALSE(ifBuilder.getIfOp().thenRegion().empty());
  EXPECT_TRUE(ifBuilder.getIfOp().elseRegion().empty());
}

TEST_F(FIRBuilderTest, genIfThenElse) {
  auto builder = getBuilder();
  auto loc = builder.getUnknownLoc();
  auto cdt = createCondition(builder);
  auto ifBuilder = builder.genIfThenElse(loc, cdt);
  EXPECT_FALSE(ifBuilder.getIfOp().thenRegion().empty());
  EXPECT_FALSE(ifBuilder.getIfOp().elseRegion().empty());
}

TEST_F(FIRBuilderTest, genIfWithThen) {
  auto builder = getBuilder();
  auto loc = builder.getUnknownLoc();
  auto cdt = createCondition(builder);
  auto ifBuilder = builder.genIfOp(loc, {}, cdt, false);
  EXPECT_FALSE(ifBuilder.getIfOp().thenRegion().empty());
  EXPECT_TRUE(ifBuilder.getIfOp().elseRegion().empty());
}

TEST_F(FIRBuilderTest, genIfWithThenAndElse) {
  auto builder = getBuilder();
  auto loc = builder.getUnknownLoc();
  auto cdt = createCondition(builder);
  auto ifBuilder = builder.genIfOp(loc, {}, cdt, true);
  EXPECT_FALSE(ifBuilder.getIfOp().thenRegion().empty());
  EXPECT_FALSE(ifBuilder.getIfOp().elseRegion().empty());
}

//===----------------------------------------------------------------------===//
// Helper functions tests
//===----------------------------------------------------------------------===//

TEST_F(FIRBuilderTest, genIsNotNull) {
  auto builder = getBuilder();
  auto loc = builder.getUnknownLoc();
  auto dummyValue =
      builder.createIntegerConstant(loc, builder.getIndexType(), 0);
  auto res = builder.genIsNotNull(loc, dummyValue);
  EXPECT_TRUE(mlir::isa<arith::CmpIOp>(res.getDefiningOp()));
  auto cmpOp = dyn_cast<arith::CmpIOp>(res.getDefiningOp());
  EXPECT_EQ(arith::CmpIPredicate::ne, cmpOp.predicate());
}

TEST_F(FIRBuilderTest, genIsNull) {
  auto builder = getBuilder();
  auto loc = builder.getUnknownLoc();
  auto dummyValue =
      builder.createIntegerConstant(loc, builder.getIndexType(), 0);
  auto res = builder.genIsNull(loc, dummyValue);
  EXPECT_TRUE(mlir::isa<arith::CmpIOp>(res.getDefiningOp()));
  auto cmpOp = dyn_cast<arith::CmpIOp>(res.getDefiningOp());
  EXPECT_EQ(arith::CmpIPredicate::eq, cmpOp.predicate());
}

TEST_F(FIRBuilderTest, createZeroConstant) {
  auto builder = getBuilder();
  auto loc = builder.getUnknownLoc();

  auto cst = builder.createNullConstant(loc);
  EXPECT_TRUE(mlir::isa<fir::ZeroOp>(cst.getDefiningOp()));
  auto zeroOp = dyn_cast<fir::ZeroOp>(cst.getDefiningOp());
  EXPECT_EQ(fir::ReferenceType::get(builder.getNoneType()),
      zeroOp.getResult().getType());
  auto idxTy = builder.getIndexType();

  cst = builder.createNullConstant(loc, idxTy);
  EXPECT_TRUE(mlir::isa<fir::ZeroOp>(cst.getDefiningOp()));
  zeroOp = dyn_cast<fir::ZeroOp>(cst.getDefiningOp());
  EXPECT_EQ(builder.getIndexType(), zeroOp.getResult().getType());
}

TEST_F(FIRBuilderTest, createRealZeroConstant) {
  auto builder = getBuilder();
  auto ctx = builder.getContext();
  auto loc = builder.getUnknownLoc();
  auto realTy = mlir::FloatType::getF64(ctx);
  auto cst = builder.createRealZeroConstant(loc, realTy);
  EXPECT_TRUE(mlir::isa<mlir::arith::ConstantOp>(cst.getDefiningOp()));
  auto cstOp = dyn_cast<mlir::arith::ConstantOp>(cst.getDefiningOp());
  EXPECT_EQ(realTy, cstOp.getType());
  EXPECT_EQ(0u, cstOp.value().cast<FloatAttr>().getValue().convertToDouble());
}

TEST_F(FIRBuilderTest, createBool) {
  auto builder = getBuilder();
  auto loc = builder.getUnknownLoc();
  auto b = builder.createBool(loc, false);
  checkIntegerConstant(b, builder.getIntegerType(1), 0);
}

TEST_F(FIRBuilderTest, getVarLenSeqTy) {
  auto builder = getBuilder();
  auto ty = builder.getVarLenSeqTy(builder.getI64Type());
  EXPECT_TRUE(ty.isa<fir::SequenceType>());
  fir::SequenceType seqTy = ty.dyn_cast<fir::SequenceType>();
  EXPECT_EQ(1u, seqTy.getDimension());
  EXPECT_TRUE(fir::unwrapSequenceType(ty).isInteger(64));
}

TEST_F(FIRBuilderTest, getNamedFunction) {
  auto builder = getBuilder();
  auto func2 = builder.getNamedFunction("func2");
  EXPECT_EQ(nullptr, func2);
  auto loc = builder.getUnknownLoc();
  func2 = builder.createFunction(
      loc, "func2", builder.getFunctionType(llvm::None, llvm::None));
  auto func2query = builder.getNamedFunction("func2");
  EXPECT_EQ(func2, func2query);
}

TEST_F(FIRBuilderTest, createGlobal1) {
  auto builder = getBuilder();
  auto loc = builder.getUnknownLoc();
  auto i64Type = IntegerType::get(builder.getContext(), 64);
  auto global = builder.createGlobal(
      loc, i64Type, "global1", builder.createInternalLinkage(), {}, true);
  EXPECT_TRUE(mlir::isa<fir::GlobalOp>(global));
  EXPECT_EQ("global1", global.sym_name());
  EXPECT_TRUE(global.constant().hasValue());
  EXPECT_EQ(i64Type, global.type());
  EXPECT_TRUE(global.linkName().hasValue());
  EXPECT_EQ(
      builder.createInternalLinkage().getValue(), global.linkName().getValue());
  EXPECT_FALSE(global.initVal().hasValue());

  auto g1 = builder.getNamedGlobal("global1");
  EXPECT_EQ(global, g1);
  auto g2 = builder.getNamedGlobal("global7");
  EXPECT_EQ(nullptr, g2);
  auto g3 = builder.getNamedGlobal("");
  EXPECT_EQ(nullptr, g3);
}

TEST_F(FIRBuilderTest, createGlobal2) {
  auto builder = getBuilder();
  auto loc = builder.getUnknownLoc();
  auto i32Type = IntegerType::get(builder.getContext(), 32);
  auto attr = builder.getIntegerAttr(i32Type, 16);
  auto global = builder.createGlobal(
      loc, i32Type, "global2", builder.createLinkOnceLinkage(), attr, false);
  EXPECT_TRUE(mlir::isa<fir::GlobalOp>(global));
  EXPECT_EQ("global2", global.sym_name());
  EXPECT_FALSE(global.constant().hasValue());
  EXPECT_EQ(i32Type, global.type());
  EXPECT_TRUE(global.initVal().hasValue());
  EXPECT_TRUE(global.initVal().getValue().isa<mlir::IntegerAttr>());
  EXPECT_EQ(
      16, global.initVal().getValue().cast<mlir::IntegerAttr>().getValue());
  EXPECT_TRUE(global.linkName().hasValue());
  EXPECT_EQ(
      builder.createLinkOnceLinkage().getValue(), global.linkName().getValue());
}

TEST_F(FIRBuilderTest, uniqueCFIdent) {
  auto str1 = fir::factory::uniqueCGIdent("", "func1");
  EXPECT_EQ("_QQ.66756E6331", str1);
  str1 = fir::factory::uniqueCGIdent("", "");
  EXPECT_EQ("_QQ.", str1);
  str1 = fir::factory::uniqueCGIdent("pr", "func1");
  EXPECT_EQ("_QQpr.66756E6331", str1);
  str1 = fir::factory::uniqueCGIdent(
      "", "longnamemorethan32characterneedshashing");
  EXPECT_EQ("_QQ.c22a886b2f30ea8c064ef1178377fc31", str1);
  str1 = fir::factory::uniqueCGIdent(
      "pr", "longnamemorethan32characterneedshashing");
  EXPECT_EQ("_QQpr.c22a886b2f30ea8c064ef1178377fc31", str1);
}

TEST_F(FIRBuilderTest, locationToLineNo) {
  auto builder = getBuilder();
  auto loc = mlir::FileLineColLoc::get(builder.getIdentifier("file1"), 10, 5);
  mlir::Value line =
      fir::factory::locationToLineNo(builder, loc, builder.getI64Type());
  checkIntegerConstant(line, builder.getI64Type(), 10);
  line = fir::factory::locationToLineNo(
      builder, builder.getUnknownLoc(), builder.getI64Type());
  checkIntegerConstant(line, builder.getI64Type(), 0);
}

TEST_F(FIRBuilderTest, hasDynamicSize) {
  auto builder = getBuilder();
  auto type = fir::CharacterType::get(builder.getContext(), 1, 16);
  EXPECT_FALSE(fir::hasDynamicSize(type));
  EXPECT_TRUE(fir::SequenceType::getUnknownExtent());
  auto seqTy = builder.getVarLenSeqTy(builder.getI64Type(), 10);
  EXPECT_TRUE(fir::hasDynamicSize(seqTy));
  EXPECT_FALSE(fir::hasDynamicSize(builder.getI64Type()));
}

TEST_F(FIRBuilderTest, locationToFilename) {
  auto builder = getBuilder();
  auto loc =
      mlir::FileLineColLoc::get(builder.getIdentifier("file1.f90"), 10, 5);
  mlir::Value locToFile = fir::factory::locationToFilename(builder, loc);
  auto addrOp = dyn_cast<fir::AddrOfOp>(locToFile.getDefiningOp());
  auto symbol = addrOp.symbol().getRootReference().getValue();
  auto global = builder.getNamedGlobal(symbol);
  auto stringLitOps = global.getRegion().front().getOps<fir::StringLitOp>();
  EXPECT_TRUE(llvm::hasSingleElement(stringLitOps));
  for (auto stringLit : stringLitOps) {
    EXPECT_EQ(10, stringLit.getSize().cast<mlir::IntegerAttr>().getValue());
    EXPECT_TRUE(stringLit.getValue().isa<StringAttr>());
    EXPECT_EQ(0,
        strcmp("file1.f90\0",
            stringLit.getValue()
                .dyn_cast<StringAttr>()
                .getValue()
                .str()
                .c_str()));
  }
}

TEST_F(FIRBuilderTest, createStringLitOp) {
  auto builder = getBuilder();
  llvm::StringRef data("mystringlitdata");
  auto loc = builder.getUnknownLoc();
  auto op = builder.createStringLitOp(loc, data);
  EXPECT_EQ(15, op.getSize().cast<mlir::IntegerAttr>().getValue());
  EXPECT_TRUE(op.getValue().isa<StringAttr>());
  EXPECT_EQ(data, op.getValue().dyn_cast<StringAttr>().getValue());
}

TEST_F(FIRBuilderTest, createStringLiteral) {
  auto builder = getBuilder();
  auto loc = builder.getUnknownLoc();
  llvm::StringRef strValue("onestringliteral");
  auto strLit = fir::factory::createStringLiteral(builder, loc, strValue);
  EXPECT_EQ(0u, strLit.rank());
  EXPECT_TRUE(strLit.getCharBox() != nullptr);
  auto *charBox = strLit.getCharBox();
  EXPECT_FALSE(fir::isArray(*charBox));
  checkIntegerConstant(charBox->getLen(), builder.getCharacterLengthType(), 16);
  auto generalGetLen = fir::getLen(strLit);
  checkIntegerConstant(generalGetLen, builder.getCharacterLengthType(), 16);
  auto addr = charBox->getBuffer();
  EXPECT_TRUE(mlir::isa<fir::AddrOfOp>(addr.getDefiningOp()));
  auto addrOp = dyn_cast<fir::AddrOfOp>(addr.getDefiningOp());
  auto symbol = addrOp.symbol().getRootReference().getValue();
  auto global = builder.getNamedGlobal(symbol);
  EXPECT_EQ(
      builder.createLinkOnceLinkage().getValue(), global.linkName().getValue());
  EXPECT_EQ(fir::CharacterType::get(builder.getContext(), 1, strValue.size()),
      global.type());

  auto stringLitOps = global.getRegion().front().getOps<fir::StringLitOp>();
  EXPECT_TRUE(llvm::hasSingleElement(stringLitOps));
  for (auto stringLit : stringLitOps) {
    EXPECT_EQ(16, stringLit.getSize().cast<mlir::IntegerAttr>().getValue());
    EXPECT_TRUE(stringLit.getValue().isa<StringAttr>());
    EXPECT_EQ(strValue, stringLit.getValue().dyn_cast<StringAttr>().getValue());
  }
}
