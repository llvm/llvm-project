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

using namespace mlir;

struct FIRBuilderTest : public testing::Test {
public:
  void SetUp() override {
    fir::support::loadDialects(context);

    llvm::ArrayRef<fir::KindTy> defs;
    fir::KindMapping kindMap(&context, defs);
    mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();

    // Set up a Module with a dummy function operation inside.
    // Set the insertion point in the function entry block.
    mlir::ModuleOp mod = builder.create<mlir::ModuleOp>(loc);
    mlir::func::FuncOp func = mlir::func::FuncOp::create(
        loc, "func1", builder.getFunctionType(llvm::None, llvm::None));
    auto *entryBlock = func.addEntryBlock();
    mod.push_back(mod);
    builder.setInsertionPointToStart(entryBlock);

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
  EXPECT_TRUE(mlir::isa<mlir::arith::ConstantOp>(value.getDefiningOp()));
  auto cstOp = dyn_cast<mlir::arith::ConstantOp>(value.getDefiningOp());
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
  EXPECT_FALSE(ifBuilder.getIfOp().getThenRegion().empty());
  EXPECT_TRUE(ifBuilder.getIfOp().getElseRegion().empty());
}

TEST_F(FIRBuilderTest, genIfThenElse) {
  auto builder = getBuilder();
  auto loc = builder.getUnknownLoc();
  auto cdt = createCondition(builder);
  auto ifBuilder = builder.genIfThenElse(loc, cdt);
  EXPECT_FALSE(ifBuilder.getIfOp().getThenRegion().empty());
  EXPECT_FALSE(ifBuilder.getIfOp().getElseRegion().empty());
}

TEST_F(FIRBuilderTest, genIfWithThen) {
  auto builder = getBuilder();
  auto loc = builder.getUnknownLoc();
  auto cdt = createCondition(builder);
  auto ifBuilder = builder.genIfOp(loc, {}, cdt, false);
  EXPECT_FALSE(ifBuilder.getIfOp().getThenRegion().empty());
  EXPECT_TRUE(ifBuilder.getIfOp().getElseRegion().empty());
}

TEST_F(FIRBuilderTest, genIfWithThenAndElse) {
  auto builder = getBuilder();
  auto loc = builder.getUnknownLoc();
  auto cdt = createCondition(builder);
  auto ifBuilder = builder.genIfOp(loc, {}, cdt, true);
  EXPECT_FALSE(ifBuilder.getIfOp().getThenRegion().empty());
  EXPECT_FALSE(ifBuilder.getIfOp().getElseRegion().empty());
}

//===----------------------------------------------------------------------===//
// Helper functions tests
//===----------------------------------------------------------------------===//

TEST_F(FIRBuilderTest, genIsNotNullAddr) {
  auto builder = getBuilder();
  auto loc = builder.getUnknownLoc();
  auto dummyValue =
      builder.createIntegerConstant(loc, builder.getIndexType(), 0);
  auto res = builder.genIsNotNullAddr(loc, dummyValue);
  EXPECT_TRUE(mlir::isa<arith::CmpIOp>(res.getDefiningOp()));
  auto cmpOp = dyn_cast<arith::CmpIOp>(res.getDefiningOp());
  EXPECT_EQ(arith::CmpIPredicate::ne, cmpOp.getPredicate());
}

TEST_F(FIRBuilderTest, genIsNullAddr) {
  auto builder = getBuilder();
  auto loc = builder.getUnknownLoc();
  auto dummyValue =
      builder.createIntegerConstant(loc, builder.getIndexType(), 0);
  auto res = builder.genIsNullAddr(loc, dummyValue);
  EXPECT_TRUE(mlir::isa<arith::CmpIOp>(res.getDefiningOp()));
  auto cmpOp = dyn_cast<arith::CmpIOp>(res.getDefiningOp());
  EXPECT_EQ(arith::CmpIPredicate::eq, cmpOp.getPredicate());
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
  EXPECT_TRUE(mlir::isa<arith::ConstantOp>(cst.getDefiningOp()));
  auto cstOp = dyn_cast<arith::ConstantOp>(cst.getDefiningOp());
  EXPECT_EQ(realTy, cstOp.getType());
  EXPECT_EQ(
      0u, cstOp.getValue().cast<FloatAttr>().getValue().convertToDouble());
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
  EXPECT_EQ("global1", global.getSymName());
  EXPECT_TRUE(global.getConstant().has_value());
  EXPECT_EQ(i64Type, global.getType());
  EXPECT_TRUE(global.getLinkName().has_value());
  EXPECT_EQ(
      builder.createInternalLinkage().getValue(), global.getLinkName().value());
  EXPECT_FALSE(global.getInitVal().has_value());

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
  EXPECT_EQ("global2", global.getSymName());
  EXPECT_FALSE(global.getConstant().has_value());
  EXPECT_EQ(i32Type, global.getType());
  EXPECT_TRUE(global.getInitVal().has_value());
  EXPECT_TRUE(global.getInitVal().value().isa<mlir::IntegerAttr>());
  EXPECT_EQ(
      16, global.getInitVal().value().cast<mlir::IntegerAttr>().getValue());
  EXPECT_TRUE(global.getLinkName().has_value());
  EXPECT_EQ(
      builder.createLinkOnceLinkage().getValue(), global.getLinkName().value());
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
  auto loc = mlir::FileLineColLoc::get(builder.getStringAttr("file1"), 10, 5);
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
      mlir::FileLineColLoc::get(builder.getStringAttr("file1.f90"), 10, 5);
  mlir::Value locToFile = fir::factory::locationToFilename(builder, loc);
  auto addrOp = dyn_cast<fir::AddrOfOp>(locToFile.getDefiningOp());
  auto symbol = addrOp.getSymbol().getRootReference().getValue();
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
  auto symbol = addrOp.getSymbol().getRootReference().getValue();
  auto global = builder.getNamedGlobal(symbol);
  EXPECT_EQ(
      builder.createLinkOnceLinkage().getValue(), global.getLinkName().value());
  EXPECT_EQ(fir::CharacterType::get(builder.getContext(), 1, strValue.size()),
      global.getType());

  auto stringLitOps = global.getRegion().front().getOps<fir::StringLitOp>();
  EXPECT_TRUE(llvm::hasSingleElement(stringLitOps));
  for (auto stringLit : stringLitOps) {
    EXPECT_EQ(16, stringLit.getSize().cast<mlir::IntegerAttr>().getValue());
    EXPECT_TRUE(stringLit.getValue().isa<StringAttr>());
    EXPECT_EQ(strValue, stringLit.getValue().dyn_cast<StringAttr>().getValue());
  }
}

TEST_F(FIRBuilderTest, allocateLocal) {
  auto builder = getBuilder();
  auto loc = builder.getUnknownLoc();
  llvm::StringRef varName = "var1";
  auto var = builder.allocateLocal(
      loc, builder.getI64Type(), "", varName, {}, {}, false);
  EXPECT_TRUE(mlir::isa<fir::AllocaOp>(var.getDefiningOp()));
  auto allocaOp = dyn_cast<fir::AllocaOp>(var.getDefiningOp());
  EXPECT_EQ(builder.getI64Type(), allocaOp.getInType());
  EXPECT_TRUE(allocaOp.getBindcName().has_value());
  EXPECT_EQ(varName, allocaOp.getBindcName().value());
  EXPECT_FALSE(allocaOp.getUniqName().has_value());
  EXPECT_FALSE(allocaOp.getPinned());
  EXPECT_EQ(0u, allocaOp.getTypeparams().size());
  EXPECT_EQ(0u, allocaOp.getShape().size());
}

static void checkShapeOp(mlir::Value shape, mlir::Value c10, mlir::Value c100) {
  EXPECT_TRUE(mlir::isa<fir::ShapeOp>(shape.getDefiningOp()));
  fir::ShapeOp op = dyn_cast<fir::ShapeOp>(shape.getDefiningOp());
  auto shapeTy = op.getType().dyn_cast<fir::ShapeType>();
  EXPECT_EQ(2u, shapeTy.getRank());
  EXPECT_EQ(2u, op.getExtents().size());
  EXPECT_EQ(c10, op.getExtents()[0]);
  EXPECT_EQ(c100, op.getExtents()[1]);
}

TEST_F(FIRBuilderTest, genShapeWithExtents) {
  auto builder = getBuilder();
  auto loc = builder.getUnknownLoc();
  auto c10 = builder.createIntegerConstant(loc, builder.getI64Type(), 10);
  auto c100 = builder.createIntegerConstant(loc, builder.getI64Type(), 100);
  llvm::SmallVector<mlir::Value> extents = {c10, c100};
  auto shape = builder.genShape(loc, extents);
  checkShapeOp(shape, c10, c100);
}

TEST_F(FIRBuilderTest, genShapeWithExtentsAndShapeShift) {
  auto builder = getBuilder();
  auto loc = builder.getUnknownLoc();
  auto c10 = builder.createIntegerConstant(loc, builder.getI64Type(), 10);
  auto c100 = builder.createIntegerConstant(loc, builder.getI64Type(), 100);
  auto c1 = builder.createIntegerConstant(loc, builder.getI64Type(), 100);
  llvm::SmallVector<mlir::Value> shifts = {c1, c1};
  llvm::SmallVector<mlir::Value> extents = {c10, c100};
  auto shape = builder.genShape(loc, shifts, extents);
  EXPECT_TRUE(mlir::isa<fir::ShapeShiftOp>(shape.getDefiningOp()));
  fir::ShapeShiftOp op = dyn_cast<fir::ShapeShiftOp>(shape.getDefiningOp());
  auto shapeTy = op.getType().dyn_cast<fir::ShapeShiftType>();
  EXPECT_EQ(2u, shapeTy.getRank());
  EXPECT_EQ(2u, op.getExtents().size());
  EXPECT_EQ(2u, op.getOrigins().size());
}

TEST_F(FIRBuilderTest, genShapeWithAbstractArrayBox) {
  auto builder = getBuilder();
  auto loc = builder.getUnknownLoc();
  auto c10 = builder.createIntegerConstant(loc, builder.getI64Type(), 10);
  auto c100 = builder.createIntegerConstant(loc, builder.getI64Type(), 100);
  llvm::SmallVector<mlir::Value> extents = {c10, c100};
  fir::AbstractArrayBox aab(extents, {});
  EXPECT_TRUE(aab.lboundsAllOne());
  auto shape = builder.genShape(loc, aab);
  checkShapeOp(shape, c10, c100);
}

TEST_F(FIRBuilderTest, readCharLen) {
  auto builder = getBuilder();
  auto loc = builder.getUnknownLoc();
  llvm::StringRef strValue("length");
  auto strLit = fir::factory::createStringLiteral(builder, loc, strValue);
  auto len = fir::factory::readCharLen(builder, loc, strLit);
  EXPECT_EQ(strLit.getCharBox()->getLen(), len);
}

TEST_F(FIRBuilderTest, getExtents) {
  auto builder = getBuilder();
  auto loc = builder.getUnknownLoc();
  llvm::StringRef strValue("length");
  auto strLit = fir::factory::createStringLiteral(builder, loc, strValue);
  auto ext = fir::factory::getExtents(loc, builder, strLit);
  EXPECT_EQ(0u, ext.size());
  auto c10 = builder.createIntegerConstant(loc, builder.getI64Type(), 10);
  auto c100 = builder.createIntegerConstant(loc, builder.getI64Type(), 100);
  llvm::SmallVector<mlir::Value> extents = {c10, c100};
  fir::SequenceType::Shape shape(2, fir::SequenceType::getUnknownExtent());
  auto arrayTy = fir::SequenceType::get(shape, builder.getI64Type());
  mlir::Value array = builder.create<fir::UndefOp>(loc, arrayTy);
  fir::ArrayBoxValue aab(array, extents, {});
  fir::ExtendedValue ex(aab);
  auto readExtents = fir::factory::getExtents(loc, builder, ex);
  EXPECT_EQ(2u, readExtents.size());
}

TEST_F(FIRBuilderTest, createZeroValue) {
  auto builder = getBuilder();
  auto loc = builder.getUnknownLoc();

  mlir::Type i64Ty = mlir::IntegerType::get(builder.getContext(), 64);
  mlir::Value zeroInt = fir::factory::createZeroValue(builder, loc, i64Ty);
  EXPECT_TRUE(zeroInt.getType() == i64Ty);
  auto cst =
      mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(zeroInt.getDefiningOp());
  EXPECT_TRUE(cst);
  auto intAttr = cst.getValue().dyn_cast<mlir::IntegerAttr>();
  EXPECT_TRUE(intAttr && intAttr.getInt() == 0);

  mlir::Type f32Ty = mlir::FloatType::getF32(builder.getContext());
  mlir::Value zeroFloat = fir::factory::createZeroValue(builder, loc, f32Ty);
  EXPECT_TRUE(zeroFloat.getType() == f32Ty);
  auto cst2 = mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(
      zeroFloat.getDefiningOp());
  EXPECT_TRUE(cst2);
  auto floatAttr = cst2.getValue().dyn_cast<mlir::FloatAttr>();
  EXPECT_TRUE(floatAttr && floatAttr.getValueAsDouble() == 0.);

  mlir::Type boolTy = mlir::IntegerType::get(builder.getContext(), 1);
  mlir::Value flaseBool = fir::factory::createZeroValue(builder, loc, boolTy);
  EXPECT_TRUE(flaseBool.getType() == boolTy);
  auto cst3 = mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(
      flaseBool.getDefiningOp());
  EXPECT_TRUE(cst3);
  auto intAttr2 = cst.getValue().dyn_cast<mlir::IntegerAttr>();
  EXPECT_TRUE(intAttr2 && intAttr2.getInt() == 0);
}

TEST_F(FIRBuilderTest, getBaseTypeOf) {
  auto builder = getBuilder();
  auto loc = builder.getUnknownLoc();

  auto makeExv = [&](mlir::Type elementType, mlir::Type arrayType)
      -> std::tuple<llvm::SmallVector<fir::ExtendedValue, 4>,
          llvm::SmallVector<fir::ExtendedValue, 4>> {
    auto ptrTyArray = fir::PointerType::get(arrayType);
    auto ptrTyScalar = fir::PointerType::get(elementType);
    auto ptrBoxTyArray = fir::BoxType::get(ptrTyArray);
    auto ptrBoxTyScalar = fir::BoxType::get(ptrTyScalar);
    auto boxRefTyArray = fir::ReferenceType::get(ptrBoxTyArray);
    auto boxRefTyScalar = fir::ReferenceType::get(ptrBoxTyScalar);
    auto boxTyArray = fir::BoxType::get(arrayType);
    auto boxTyScalar = fir::BoxType::get(elementType);

    auto ptrValArray = builder.create<fir::UndefOp>(loc, ptrTyArray);
    auto ptrValScalar = builder.create<fir::UndefOp>(loc, ptrTyScalar);
    auto boxRefValArray = builder.create<fir::UndefOp>(loc, boxRefTyArray);
    auto boxRefValScalar = builder.create<fir::UndefOp>(loc, boxRefTyScalar);
    auto boxValArray = builder.create<fir::UndefOp>(loc, boxTyArray);
    auto boxValScalar = builder.create<fir::UndefOp>(loc, boxTyScalar);

    llvm::SmallVector<fir::ExtendedValue, 4> scalars;
    scalars.emplace_back(fir::UnboxedValue(ptrValScalar));
    scalars.emplace_back(fir::BoxValue(boxValScalar));
    scalars.emplace_back(
        fir::MutableBoxValue(boxRefValScalar, mlir::ValueRange(), {}));

    llvm::SmallVector<fir::ExtendedValue, 4> arrays;
    auto extent = builder.create<fir::UndefOp>(loc, builder.getIndexType());
    llvm::SmallVector<mlir::Value> extents(
        arrayType.dyn_cast<fir::SequenceType>().getDimension(),
        extent.getResult());
    arrays.emplace_back(fir::ArrayBoxValue(ptrValArray, extents));
    arrays.emplace_back(fir::BoxValue(boxValArray));
    arrays.emplace_back(
        fir::MutableBoxValue(boxRefValArray, mlir::ValueRange(), {}));
    return {scalars, arrays};
  };

  auto f32Ty = mlir::FloatType::getF32(builder.getContext());
  mlir::Type f32SeqTy = builder.getVarLenSeqTy(f32Ty);
  auto [f32Scalars, f32Arrays] = makeExv(f32Ty, f32SeqTy);
  for (const auto &scalar : f32Scalars) {
    EXPECT_EQ(fir::getBaseTypeOf(scalar), f32Ty);
    EXPECT_EQ(fir::getElementTypeOf(scalar), f32Ty);
    EXPECT_FALSE(fir::isDerivedWithLenParameters(scalar));
  }
  for (const auto &array : f32Arrays) {
    EXPECT_EQ(fir::getBaseTypeOf(array), f32SeqTy);
    EXPECT_EQ(fir::getElementTypeOf(array), f32Ty);
    EXPECT_FALSE(fir::isDerivedWithLenParameters(array));
  }

  auto derivedWithLengthTy =
      fir::RecordType::get(builder.getContext(), "derived_test");

  llvm::SmallVector<std::pair<std::string, mlir::Type>> parameters;
  llvm::SmallVector<std::pair<std::string, mlir::Type>> components;
  parameters.emplace_back("p1", builder.getI64Type());
  components.emplace_back("c1", f32Ty);
  derivedWithLengthTy.finalize(parameters, components);
  mlir::Type derivedWithLengthSeqTy =
      builder.getVarLenSeqTy(derivedWithLengthTy);
  auto [derivedWithLengthScalars, derivedWithLengthArrays] =
      makeExv(derivedWithLengthTy, derivedWithLengthSeqTy);
  for (const auto &scalar : derivedWithLengthScalars) {
    EXPECT_EQ(fir::getBaseTypeOf(scalar), derivedWithLengthTy);
    EXPECT_EQ(fir::getElementTypeOf(scalar), derivedWithLengthTy);
    EXPECT_TRUE(fir::isDerivedWithLenParameters(scalar));
  }
  for (const auto &array : derivedWithLengthArrays) {
    EXPECT_EQ(fir::getBaseTypeOf(array), derivedWithLengthSeqTy);
    EXPECT_EQ(fir::getElementTypeOf(array), derivedWithLengthTy);
    EXPECT_TRUE(fir::isDerivedWithLenParameters(array));
  }
}

TEST_F(FIRBuilderTest, genArithFastMath) {
  auto builder = getBuilder();
  auto ctx = builder.getContext();
  auto loc = builder.getUnknownLoc();

  auto realTy = mlir::FloatType::getF32(ctx);
  auto arg = builder.create<fir::UndefOp>(loc, realTy);

  // Test that FastMathFlags is 'none' by default.
  mlir::Operation *op1 = builder.create<mlir::arith::AddFOp>(loc, arg, arg);
  auto op1_fmi =
      mlir::dyn_cast_or_null<mlir::arith::ArithFastMathInterface>(op1);
  EXPECT_TRUE(op1_fmi);
  auto op1_fmf = op1_fmi.getFastMathFlagsAttr().getValue();
  EXPECT_EQ(op1_fmf, arith::FastMathFlags::none);

  // Test that the builder is copied properly.
  fir::FirOpBuilder builder_copy(builder);

  arith::FastMathFlags FMF1 =
      arith::FastMathFlags::contract | arith::FastMathFlags::reassoc;
  builder.setFastMathFlags(FMF1);
  arith::FastMathFlags FMF2 =
      arith::FastMathFlags::nnan | arith::FastMathFlags::ninf;
  builder_copy.setFastMathFlags(FMF2);

  // Modifying FastMathFlags for the copy must not affect the original builder.
  mlir::Operation *op2 = builder.create<mlir::arith::AddFOp>(loc, arg, arg);
  auto op2_fmi =
      mlir::dyn_cast_or_null<mlir::arith::ArithFastMathInterface>(op2);
  EXPECT_TRUE(op2_fmi);
  auto op2_fmf = op2_fmi.getFastMathFlagsAttr().getValue();
  EXPECT_EQ(op2_fmf, FMF1);

  // Modifying FastMathFlags for the original builder must not affect the copy.
  mlir::Operation *op3 =
      builder_copy.create<mlir::arith::AddFOp>(loc, arg, arg);
  auto op3_fmi =
      mlir::dyn_cast_or_null<mlir::arith::ArithFastMathInterface>(op3);
  EXPECT_TRUE(op3_fmi);
  auto op3_fmf = op3_fmi.getFastMathFlagsAttr().getValue();
  EXPECT_EQ(op3_fmf, FMF2);

  // Test that the builder copy inherits FastMathFlags from the original.
  fir::FirOpBuilder builder_copy2(builder);

  mlir::Operation *op4 =
      builder_copy2.create<mlir::arith::AddFOp>(loc, arg, arg);
  auto op4_fmi =
      mlir::dyn_cast_or_null<mlir::arith::ArithFastMathInterface>(op4);
  EXPECT_TRUE(op4_fmi);
  auto op4_fmf = op4_fmi.getFastMathFlagsAttr().getValue();
  EXPECT_EQ(op4_fmf, FMF1);
}
