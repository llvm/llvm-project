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
    builder = std::make_unique<mlir::OpBuilder>(&context);
    mlir::Location loc = builder->getUnknownLoc();

    // Set up a Module with a dummy function operation inside.
    // Set the insertion point in the function entry block.
    mlir::ModuleOp mod = builder->create<mlir::ModuleOp>(loc);
    mlir::func::FuncOp func =
        mlir::func::FuncOp::create(loc, "fortran_variable_tests",
            builder->getFunctionType(llvm::None, llvm::None));
    auto *entryBlock = func.addEntryBlock();
    mod.push_back(mod);
    builder->setInsertionPointToStart(entryBlock);
  }

  mlir::Location getLoc() { return builder->getUnknownLoc(); }
  mlir::Value createConstant(std::int64_t cst) {
    mlir::Type indexType = builder->getIndexType();
    return builder->create<mlir::arith::ConstantOp>(
        getLoc(), indexType, builder->getIntegerAttr(indexType, cst));
  }

  mlir::Value createShape(llvm::ArrayRef<mlir::Value> extents) {
    mlir::Type shapeType = fir::ShapeType::get(&context, extents.size());
    return builder->create<fir::ShapeOp>(getLoc(), shapeType, extents);
  }
  mlir::MLIRContext context;
  std::unique_ptr<mlir::OpBuilder> builder;
};

TEST_F(FortranVariableTest, SimpleScalar) {
  mlir::Location loc = getLoc();
  mlir::Type eleType = mlir::FloatType::getF32(&context);
  mlir::Value addr = builder->create<fir::AllocaOp>(loc, eleType);
  auto name = mlir::StringAttr::get(&context, "x");
  auto declare = builder->create<fir::DeclareOp>(loc, addr.getType(), addr,
      /*shape=*/mlir::Value{}, /*typeParams=*/llvm::None, name,
      /*fortran_attrs=*/fir::FortranVariableFlagsAttr{});

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
  mlir::Location loc = getLoc();
  mlir::Type eleType = fir::CharacterType::getUnknownLen(&context, 4);
  mlir::Value len = createConstant(42);
  llvm::SmallVector<mlir::Value> typeParams{len};
  mlir::Value addr = builder->create<fir::AllocaOp>(
      loc, eleType, /*pinned=*/false, typeParams);
  auto name = mlir::StringAttr::get(&context, "x");
  auto declare = builder->create<fir::DeclareOp>(loc, addr.getType(), addr,
      /*shape=*/mlir::Value{}, typeParams, name,
      /*fortran_attrs=*/fir::FortranVariableFlagsAttr{});

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
  mlir::Location loc = getLoc();
  mlir::Type eleType = mlir::FloatType::getF32(&context);
  llvm::SmallVector<mlir::Value> extents{
      createConstant(10), createConstant(20), createConstant(30)};
  fir::SequenceType::Shape typeShape(
      extents.size(), fir::SequenceType::getUnknownExtent());
  mlir::Type seqTy = fir::SequenceType::get(typeShape, eleType);
  mlir::Value addr = builder->create<fir::AllocaOp>(
      loc, seqTy, /*pinned=*/false, /*typeParams=*/llvm::None, extents);
  mlir::Value shape = createShape(extents);
  auto name = mlir::StringAttr::get(&context, "x");
  auto declare = builder->create<fir::DeclareOp>(loc, addr.getType(), addr,
      shape, /*typeParams*/ llvm::None, name,
      /*fortran_attrs=*/fir::FortranVariableFlagsAttr{});

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
  mlir::Location loc = getLoc();
  mlir::Type eleType = fir::CharacterType::getUnknownLen(&context, 4);
  mlir::Value len = createConstant(42);
  llvm::SmallVector<mlir::Value> typeParams{len};
  llvm::SmallVector<mlir::Value> extents{
      createConstant(10), createConstant(20), createConstant(30)};
  fir::SequenceType::Shape typeShape(
      extents.size(), fir::SequenceType::getUnknownExtent());
  mlir::Type seqTy = fir::SequenceType::get(typeShape, eleType);
  mlir::Value addr = builder->create<fir::AllocaOp>(
      loc, seqTy, /*pinned=*/false, typeParams, extents);
  mlir::Value shape = createShape(extents);
  auto name = mlir::StringAttr::get(&context, "x");
  auto declare = builder->create<fir::DeclareOp>(loc, addr.getType(), addr,
      shape, typeParams, name,
      /*fortran_attrs=*/fir::FortranVariableFlagsAttr{});

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
