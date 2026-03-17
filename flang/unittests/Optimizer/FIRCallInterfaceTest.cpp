//===- FIRCallInterfaceTest.cpp - fir::CallOp setCalleeFromCallable tests -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for CallOpInterface on fir::CallOp.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Support/InitFIR.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Casting.h"

using namespace mlir;

static bool isSymbolRef(mlir::CallInterfaceCallable callable) {
  return llvm::isa<SymbolRefAttr>(callable);
}
static bool isValue(mlir::CallInterfaceCallable callable) {
  return llvm::isa<Value>(callable);
}

struct FIRCallInterfaceTest : public testing::Test {
  void SetUp() override { fir::support::loadDialects(context); }

  MLIRContext context;
};

TEST_F(FIRCallInterfaceTest, setCalleeFromCallable_directToDirect) {
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  ModuleOp module = ModuleOp::create(builder, loc);
  builder.setInsertionPointToStart(module.getBody());

  auto funcType = builder.getFunctionType({}, {});
  auto func = func::FuncOp::create(builder, loc, "target", funcType);
  func.setPrivate();
  func.getBody().push_back(new Block);
  builder.setInsertionPointToStart(&func.getBody().front());
  func::ReturnOp::create(builder, loc);
  builder.setInsertionPointToStart(module.getBody());

  // Direct call: fir.call @target()
  auto callTargetRef = FlatSymbolRefAttr::get(&context, "target");
  auto callOp = fir::CallOp::create(
      builder, loc, callTargetRef, llvm::ArrayRef<mlir::Type>{}, ValueRange{});
  ASSERT_TRUE(isSymbolRef(callOp.getCallableForCallee()));
  EXPECT_EQ(callOp.getNumOperands(), 0u);

  // Change to another symbol; should remain direct with no extra operand.
  auto newCallTargetRef = FlatSymbolRefAttr::get(&context, "other");
  callOp.setCalleeFromCallable(newCallTargetRef);

  EXPECT_TRUE(isSymbolRef(callOp.getCallableForCallee()));
  EXPECT_EQ(llvm::cast<SymbolRefAttr>(callOp.getCallableForCallee())
                .getRootReference()
                .getValue(),
      "other");
  EXPECT_EQ(callOp.getNumOperands(), 0u);
  EXPECT_TRUE(callOp->getAttr(fir::CallOp::getCalleeAttrNameStr()));
}

TEST_F(FIRCallInterfaceTest, setCalleeFromCallable_indirectToDirect) {
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  ModuleOp module = ModuleOp::create(builder, loc);
  builder.setInsertionPointToStart(module.getBody());

  auto funcType = builder.getFunctionType({}, {});
  // Container has one argument: procedure pointer () -> ()
  auto containerType = builder.getFunctionType({funcType}, {});
  auto func = func::FuncOp::create(builder, loc, "container", containerType);
  func.setPrivate();
  Block *block = func.addEntryBlock();
  builder.setInsertionPointToStart(block);

  // Indirect call: fir.call %arg0()
  Value callTargetValue = block->getArgument(0);
  auto callOp = fir::CallOp::create(builder, loc, SymbolRefAttr{},
      llvm::ArrayRef<mlir::Type>{}, ValueRange{callTargetValue});
  ASSERT_TRUE(isValue(callOp.getCallableForCallee()));
  EXPECT_EQ(callOp.getNumOperands(), 1u);
  EXPECT_FALSE(callOp->getAttr(fir::CallOp::getCalleeAttrNameStr()));

  // Switch to direct call; operand 0 must be removed.
  auto callTargetRef = FlatSymbolRefAttr::get(&context, "direct_target");
  callOp.setCalleeFromCallable(callTargetRef);

  EXPECT_TRUE(isSymbolRef(callOp.getCallableForCallee()));
  EXPECT_EQ(llvm::cast<SymbolRefAttr>(callOp.getCallableForCallee())
                .getRootReference()
                .getValue(),
      "direct_target");
  EXPECT_EQ(callOp.getNumOperands(), 0u);
  EXPECT_TRUE(callOp->getAttr(fir::CallOp::getCalleeAttrNameStr()));
}

TEST_F(FIRCallInterfaceTest, setCalleeFromCallable_directToIndirect) {
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  ModuleOp module = ModuleOp::create(builder, loc);
  builder.setInsertionPointToStart(module.getBody());

  auto funcType = builder.getFunctionType({}, {});
  auto containerType = builder.getFunctionType({funcType}, {});
  auto func = func::FuncOp::create(builder, loc, "container", containerType);
  func.setPrivate();
  Block *block = func.addEntryBlock();
  builder.setInsertionPointToStart(block);

  // Direct call first
  auto callTargetRef = FlatSymbolRefAttr::get(&context, "target");
  auto callOp = fir::CallOp::create(
      builder, loc, callTargetRef, llvm::ArrayRef<mlir::Type>{}, ValueRange{});
  ASSERT_TRUE(isSymbolRef(callOp.getCallableForCallee()));
  EXPECT_EQ(callOp.getNumOperands(), 0u);

  // Switch to indirect; attribute must be unset, operand 0 set.
  Value callTargetValue = block->getArgument(0);
  callOp.setCalleeFromCallable(callTargetValue);

  EXPECT_TRUE(isValue(callOp.getCallableForCallee()));
  EXPECT_EQ(callOp.getNumOperands(), 1u);
  EXPECT_EQ(callOp.getOperand(0), callTargetValue);
  EXPECT_FALSE(callOp->getAttr(fir::CallOp::getCalleeAttrNameStr()));
}

TEST_F(FIRCallInterfaceTest, setCalleeFromCallable_indirectToIndirect) {
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  ModuleOp module = ModuleOp::create(builder, loc);
  builder.setInsertionPointToStart(module.getBody());

  auto funcType = builder.getFunctionType({}, {});
  // Container has two arguments: procedure pointers () -> ()
  auto containerType = builder.getFunctionType({funcType, funcType}, {});
  auto func = func::FuncOp::create(builder, loc, "container", containerType);
  func.setPrivate();
  Block *block = func.addEntryBlock();
  builder.setInsertionPointToStart(block);

  Value callTarget0 = block->getArgument(0);
  Value callTarget1 = block->getArgument(1);

  // Indirect call: fir.call %arg0()
  auto callOp = fir::CallOp::create(builder, loc, SymbolRefAttr{},
      llvm::ArrayRef<mlir::Type>{}, ValueRange{callTarget0});
  ASSERT_TRUE(isValue(callOp.getCallableForCallee()));
  EXPECT_EQ(callOp.getNumOperands(), 1u);
  EXPECT_EQ(callOp.getOperand(0), callTarget0);

  // Switch to other indirect call target; should remain indirect, operand 0
  // updated.
  callOp.setCalleeFromCallable(callTarget1);

  EXPECT_TRUE(isValue(callOp.getCallableForCallee()));
  EXPECT_EQ(callOp.getNumOperands(), 1u);
  EXPECT_EQ(callOp.getOperand(0), callTarget1);
  EXPECT_FALSE(callOp->getAttr(fir::CallOp::getCalleeAttrNameStr()));
}
