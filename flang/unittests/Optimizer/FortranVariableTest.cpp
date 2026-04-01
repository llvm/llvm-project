//===- FortranVariableTest.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Support/InitFIR.h"

struct FortranVariableTest : public testing::Test {
public:
  void SetUp() {
    fir::support::loadDialects(context);
    builder = std::make_unique<aiir::OpBuilder>(&context);
    aiir::Location loc = builder->getUnknownLoc();

    // Set up a Module with a dummy function operation inside.
    // Set the insertion point in the function entry block.
    moduleOp = aiir::ModuleOp::create(*builder, loc);
    builder->setInsertionPointToStart(moduleOp->getBody());
    aiir::func::FuncOp func = aiir::func::FuncOp::create(*builder, loc,
        "fortran_variable_tests", builder->getFunctionType({}, {}));
    auto *entryBlock = func.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
  }

  aiir::Location getLoc() { return builder->getUnknownLoc(); }
  aiir::Value createConstant(std::int64_t cst) {
    aiir::Type indexType = builder->getIndexType();
    return aiir::arith::ConstantOp::create(
        *builder, getLoc(), indexType, builder->getIntegerAttr(indexType, cst));
  }

  aiir::Value createShape(llvm::ArrayRef<aiir::Value> extents) {
    return fir::ShapeOp::create(*builder, getLoc(), extents);
  }
  aiir::AIIRContext context;
  std::unique_ptr<aiir::OpBuilder> builder;
  aiir::OwningOpRef<aiir::ModuleOp> moduleOp;
};

TEST_F(FortranVariableTest, SimpleScalar) {
  aiir::Location loc = getLoc();
  aiir::Type eleType = aiir::Float32Type::get(&context);
  aiir::Value addr = fir::AllocaOp::create(*builder, loc, eleType);
  auto name = aiir::StringAttr::get(&context, "x");
  auto declare = fir::DeclareOp::create(*builder, loc, addr.getType(), addr,
      /*shape=*/aiir::Value{}, /*typeParams=*/aiir::ValueRange{},
      /*dummy_scope=*/nullptr, /*storage=*/nullptr, /*storage_offset=*/0, name,
      /*fortran_attrs=*/fir::FortranVariableFlagsAttr{},
      /*data_attr=*/cuf::DataAttributeAttr{},
      /*dummy_arg_no=*/aiir::IntegerAttr{});

  fir::FortranVariableOpInterface fortranVariable = declare;
  EXPECT_FALSE(fortranVariable.isArray());
  EXPECT_FALSE(fortranVariable.isCharacter());
  EXPECT_FALSE(fortranVariable.isPointer());
  EXPECT_FALSE(fortranVariable.isAllocatable());
  EXPECT_FALSE(fortranVariable.hasExplicitCharLen());
  EXPECT_EQ(fortranVariable.getElementType(), eleType);
  EXPECT_EQ(fortranVariable.getElementOrSequenceType(),
      fortranVariable.getElementType());
  EXPECT_NE(fortranVariable.getBase(), addr);
  EXPECT_EQ(fortranVariable.getBase().getType(), addr.getType());
}

TEST_F(FortranVariableTest, CharacterScalar) {
  aiir::Location loc = getLoc();
  aiir::Type eleType = fir::CharacterType::getUnknownLen(&context, 4);
  aiir::Value len = createConstant(42);
  llvm::SmallVector<aiir::Value> typeParams{len};
  aiir::Value addr = fir::AllocaOp::create(
      *builder, loc, eleType, /*pinned=*/false, typeParams);
  auto name = aiir::StringAttr::get(&context, "x");
  auto declare = fir::DeclareOp::create(*builder, loc, addr.getType(), addr,
      /*shape=*/aiir::Value{}, typeParams, /*dummy_scope=*/nullptr,
      /*storage=*/nullptr, /*storage_offset=*/0, name,
      /*fortran_attrs=*/fir::FortranVariableFlagsAttr{},
      /*data_attr=*/cuf::DataAttributeAttr{},
      /*dummy_arg_no=*/aiir::IntegerAttr{});

  fir::FortranVariableOpInterface fortranVariable = declare;
  EXPECT_FALSE(fortranVariable.isArray());
  EXPECT_TRUE(fortranVariable.isCharacter());
  EXPECT_FALSE(fortranVariable.isPointer());
  EXPECT_FALSE(fortranVariable.isAllocatable());
  EXPECT_TRUE(fortranVariable.hasExplicitCharLen());
  EXPECT_EQ(fortranVariable.getElementType(), eleType);
  EXPECT_EQ(fortranVariable.getElementOrSequenceType(),
      fortranVariable.getElementType());
  EXPECT_NE(fortranVariable.getBase(), addr);
  EXPECT_EQ(fortranVariable.getBase().getType(), addr.getType());
  EXPECT_EQ(fortranVariable.getExplicitCharLen(), len);
}

TEST_F(FortranVariableTest, SimpleArray) {
  aiir::Location loc = getLoc();
  aiir::Type eleType = aiir::Float32Type::get(&context);
  llvm::SmallVector<aiir::Value> extents{
      createConstant(10), createConstant(20), createConstant(30)};
  fir::SequenceType::Shape typeShape(
      extents.size(), fir::SequenceType::getUnknownExtent());
  aiir::Type seqTy = fir::SequenceType::get(typeShape, eleType);
  aiir::Value addr = fir::AllocaOp::create(*builder, loc, seqTy,
      /*pinned=*/false, /*typeParams=*/aiir::ValueRange{}, extents);
  aiir::Value shape = createShape(extents);
  auto name = aiir::StringAttr::get(&context, "x");
  auto declare = fir::DeclareOp::create(*builder, loc, addr.getType(), addr,
      shape, /*typeParams=*/aiir::ValueRange{}, /*dummy_scope=*/nullptr,
      /*storage=*/nullptr, /*storage_offset=*/0, name,
      /*fortran_attrs=*/fir::FortranVariableFlagsAttr{},
      /*data_attr=*/cuf::DataAttributeAttr{},
      /*dummy_arg_no=*/aiir::IntegerAttr{});

  fir::FortranVariableOpInterface fortranVariable = declare;
  EXPECT_TRUE(fortranVariable.isArray());
  EXPECT_FALSE(fortranVariable.isCharacter());
  EXPECT_FALSE(fortranVariable.isPointer());
  EXPECT_FALSE(fortranVariable.isAllocatable());
  EXPECT_FALSE(fortranVariable.hasExplicitCharLen());
  EXPECT_EQ(fortranVariable.getElementType(), eleType);
  EXPECT_EQ(fortranVariable.getElementOrSequenceType(), seqTy);
  EXPECT_NE(fortranVariable.getBase(), addr);
  EXPECT_EQ(fortranVariable.getBase().getType(), addr.getType());
}

TEST_F(FortranVariableTest, CharacterArray) {
  aiir::Location loc = getLoc();
  aiir::Type eleType = fir::CharacterType::getUnknownLen(&context, 4);
  aiir::Value len = createConstant(42);
  llvm::SmallVector<aiir::Value> typeParams{len};
  llvm::SmallVector<aiir::Value> extents{
      createConstant(10), createConstant(20), createConstant(30)};
  fir::SequenceType::Shape typeShape(
      extents.size(), fir::SequenceType::getUnknownExtent());
  aiir::Type seqTy = fir::SequenceType::get(typeShape, eleType);
  aiir::Value addr = fir::AllocaOp::create(
      *builder, loc, seqTy, /*pinned=*/false, typeParams, extents);
  aiir::Value shape = createShape(extents);
  auto name = aiir::StringAttr::get(&context, "x");
  auto declare = fir::DeclareOp::create(*builder, loc, addr.getType(), addr,
      shape, typeParams, /*dummy_scope=*/nullptr, /*storage=*/nullptr,
      /*storage_offset=*/0, name,
      /*fortran_attrs=*/fir::FortranVariableFlagsAttr{},
      /*data_attr=*/cuf::DataAttributeAttr{},
      /*dummy_arg_no=*/aiir::IntegerAttr{});

  fir::FortranVariableOpInterface fortranVariable = declare;
  EXPECT_TRUE(fortranVariable.isArray());
  EXPECT_TRUE(fortranVariable.isCharacter());
  EXPECT_FALSE(fortranVariable.isPointer());
  EXPECT_FALSE(fortranVariable.isAllocatable());
  EXPECT_TRUE(fortranVariable.hasExplicitCharLen());
  EXPECT_EQ(fortranVariable.getElementType(), eleType);
  EXPECT_EQ(fortranVariable.getElementOrSequenceType(), seqTy);
  EXPECT_NE(fortranVariable.getBase(), addr);
  EXPECT_EQ(fortranVariable.getBase().getType(), addr.getType());
  EXPECT_EQ(fortranVariable.getExplicitCharLen(), len);
}
