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

namespace {

static bool isSymbolRef(mlir::CallInterfaceCallable callable) {
  return llvm::isa<SymbolRefAttr>(callable);
}
static bool isValue(mlir::CallInterfaceCallable callable) {
  return llvm::isa<Value>(callable);
}

/// Creates a module and a function with entry block. Builder insertion point is
/// set to the block start. Returns (func, block) so tests can create calls in
/// the block and use block arguments as callee/args.
std::pair<func::FuncOp, Block *> createModuleWithFunction(
    OpBuilder &builder, Location loc, StringRef name, FunctionType funcType) {
  ModuleOp module = ModuleOp::create(builder, loc);
  builder.setInsertionPointToStart(module.getBody());
  auto func = func::FuncOp::create(builder, loc, name, funcType);
  func.setPrivate();
  Block *block = func.addEntryBlock();
  builder.setInsertionPointToStart(block);
  return {func, block};
}

} // namespace

struct FIRCallInterfaceTest : public testing::Test {
  void SetUp() override { fir::support::loadDialects(context); }

  MLIRContext context;
};

TEST_F(FIRCallInterfaceTest, setCalleeFromCallable_directToDirect) {
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto funcType = builder.getFunctionType({}, {});
  (void)createModuleWithFunction(builder, loc, "target", funcType);

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
  auto funcType = builder.getFunctionType({}, {});
  auto containerType = builder.getFunctionType({funcType}, {});
  auto [func, block] =
      createModuleWithFunction(builder, loc, "container", containerType);

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
  auto funcType = builder.getFunctionType({}, {});
  auto containerType = builder.getFunctionType({funcType}, {});
  auto [func, block] =
      createModuleWithFunction(builder, loc, "container", containerType);

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
  auto funcType = builder.getFunctionType({}, {});
  auto containerType = builder.getFunctionType({funcType, funcType}, {});
  auto [func, block] =
      createModuleWithFunction(builder, loc, "container", containerType);

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

TEST_F(FIRCallInterfaceTest, setCalleeFromCallable_directToIndirect_withArgs) {
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto i32Ty = builder.getI32Type();
  auto funcType = builder.getFunctionType({i32Ty}, {});
  auto containerType = builder.getFunctionType({funcType, i32Ty}, {});
  auto [func, block] =
      createModuleWithFunction(builder, loc, "container", containerType);
  Value calleeVal = block->getArgument(0);
  Value argVal = block->getArgument(1);

  // Direct call with one argument: fir.call @target(%arg)
  auto callTargetRef = FlatSymbolRefAttr::get(&context, "target");
  auto callOp = fir::CallOp::create(builder, loc, callTargetRef,
      llvm::ArrayRef<mlir::Type>{}, ValueRange{argVal});
  ASSERT_TRUE(isSymbolRef(callOp.getCallableForCallee()));
  EXPECT_EQ(callOp.getNumOperands(), 1u);
  EXPECT_EQ(callOp.getOperand(0), argVal);

  // Switch to indirect; callee must be inserted at 0, arg preserved.
  callOp.setCalleeFromCallable(calleeVal);

  EXPECT_TRUE(isValue(callOp.getCallableForCallee()));
  EXPECT_EQ(callOp.getNumOperands(), 2u);
  EXPECT_EQ(callOp.getOperand(0), calleeVal);
  EXPECT_EQ(callOp.getOperand(1), argVal);
  EXPECT_FALSE(callOp->getAttr(fir::CallOp::getCalleeAttrNameStr()));
}

TEST_F(FIRCallInterfaceTest, setCalleeFromCallable_indirectToDirect_withArgs) {
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto i32Ty = builder.getI32Type();
  auto funcType = builder.getFunctionType({i32Ty}, {});
  auto containerType = builder.getFunctionType({funcType, i32Ty}, {});
  auto [func, block] =
      createModuleWithFunction(builder, loc, "container", containerType);
  Value calleeVal = block->getArgument(0);
  Value argVal = block->getArgument(1);

  // Indirect call with one argument: fir.call %callee(%arg)
  auto callOp = fir::CallOp::create(builder, loc, SymbolRefAttr{},
      llvm::ArrayRef<mlir::Type>{}, ValueRange{calleeVal, argVal});
  ASSERT_TRUE(isValue(callOp.getCallableForCallee()));
  EXPECT_EQ(callOp.getNumOperands(), 2u);
  EXPECT_EQ(callOp.getOperand(0), calleeVal);
  EXPECT_EQ(callOp.getOperand(1), argVal);

  // Switch to direct; callee operand must be removed, arg preserved.
  auto callTargetRef = FlatSymbolRefAttr::get(&context, "direct_target");
  callOp.setCalleeFromCallable(callTargetRef);

  EXPECT_TRUE(isSymbolRef(callOp.getCallableForCallee()));
  EXPECT_EQ(callOp.getNumOperands(), 1u);
  EXPECT_EQ(callOp.getOperand(0), argVal);
  EXPECT_TRUE(callOp->getAttr(fir::CallOp::getCalleeAttrNameStr()));
}

TEST_F(
    FIRCallInterfaceTest, setCalleeFromCallable_indirectToIndirect_withArgs) {
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto i32Ty = builder.getI32Type();
  auto funcType = builder.getFunctionType({i32Ty}, {});
  auto containerType = builder.getFunctionType({funcType, funcType, i32Ty}, {});
  auto [func, block] =
      createModuleWithFunction(builder, loc, "container", containerType);
  Value callee0 = block->getArgument(0);
  Value callee1 = block->getArgument(1);
  Value argVal = block->getArgument(2);

  // Indirect call with one argument: fir.call %callee0(%arg)
  auto callOp = fir::CallOp::create(builder, loc, SymbolRefAttr{},
      llvm::ArrayRef<mlir::Type>{}, ValueRange{callee0, argVal});
  ASSERT_TRUE(isValue(callOp.getCallableForCallee()));
  EXPECT_EQ(callOp.getNumOperands(), 2u);
  EXPECT_EQ(callOp.getOperand(0), callee0);
  EXPECT_EQ(callOp.getOperand(1), argVal);

  // Switch to other indirect callee; operand 0 updated, arg preserved.
  callOp.setCalleeFromCallable(callee1);

  EXPECT_TRUE(isValue(callOp.getCallableForCallee()));
  EXPECT_EQ(callOp.getNumOperands(), 2u);
  EXPECT_EQ(callOp.getOperand(0), callee1);
  EXPECT_EQ(callOp.getOperand(1), argVal);
  EXPECT_FALSE(callOp->getAttr(fir::CallOp::getCalleeAttrNameStr()));
}

static ArrayAttr makeArgAttrs(
    MLIRContext *ctx, llvm::ArrayRef<DictionaryAttr> dicts) {
  llvm::SmallVector<Attribute> attrs(dicts.begin(), dicts.end());
  return ArrayAttr::get(ctx, attrs);
}

static DictionaryAttr makeTestArgDict(MLIRContext *ctx, StringRef value) {
  return DictionaryAttr::get(ctx,
      {NamedAttribute(
          StringAttr::get(ctx, "test.attr"), StringAttr::get(ctx, value))});
}

TEST_F(
    FIRCallInterfaceTest, setCalleeFromCallable_directToDirect_withArgAttrs) {
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto i32Ty = builder.getI32Type();
  auto funcType = builder.getFunctionType({i32Ty}, {});
  auto containerType = builder.getFunctionType({funcType, i32Ty}, {});
  auto [func, block] =
      createModuleWithFunction(builder, loc, "container", containerType);
  Value argVal = block->getArgument(1);

  // Direct call with one argument and arg_attrs.
  auto callTargetRef = FlatSymbolRefAttr::get(&context, "target");
  auto callOp = fir::CallOp::create(builder, loc, callTargetRef,
      llvm::ArrayRef<mlir::Type>{}, ValueRange{argVal});
  callOp->setAttr(callOp.getArgAttrsAttrName(),
      makeArgAttrs(&context, {makeTestArgDict(&context, "arg0")}));
  ASSERT_TRUE(isSymbolRef(callOp.getCallableForCallee()));
  EXPECT_EQ(callOp.getNumOperands(), 1u);

  // Switch to another direct callee
  auto newCallTargetRef = FlatSymbolRefAttr::get(&context, "other_target");
  callOp.setCalleeFromCallable(newCallTargetRef);

  EXPECT_TRUE(isSymbolRef(callOp.getCallableForCallee()));
  EXPECT_EQ(llvm::cast<SymbolRefAttr>(callOp.getCallableForCallee())
                .getRootReference()
                .getValue(),
      "other_target");
  EXPECT_EQ(callOp.getNumOperands(), 1u);
  EXPECT_EQ(callOp.getOperand(0), argVal);
  ArrayAttr argAttrs = callOp.getArgAttrsAttr();
  ASSERT_TRUE(argAttrs);
  ASSERT_EQ(argAttrs.size(), 1u);
  EXPECT_EQ(llvm::cast<DictionaryAttr>(argAttrs[0]).get("test.attr"),
      StringAttr::get(&context, "arg0"));
}

TEST_F(
    FIRCallInterfaceTest, setCalleeFromCallable_directToIndirect_withArgAttrs) {
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto i32Ty = builder.getI32Type();
  auto funcType = builder.getFunctionType({i32Ty}, {});
  auto containerType = builder.getFunctionType({funcType, i32Ty}, {});
  auto [func, block] =
      createModuleWithFunction(builder, loc, "container", containerType);
  Value calleeVal = block->getArgument(0);
  Value argVal = block->getArgument(1);

  // Direct call with one argument and arg_attrs for that argument.
  auto callTargetRef = FlatSymbolRefAttr::get(&context, "target");
  auto callOp = fir::CallOp::create(builder, loc, callTargetRef,
      llvm::ArrayRef<mlir::Type>{}, ValueRange{argVal});
  callOp->setAttr(callOp.getArgAttrsAttrName(),
      makeArgAttrs(&context, {makeTestArgDict(&context, "arg0")}));
  ASSERT_TRUE(isSymbolRef(callOp.getCallableForCallee()));
  EXPECT_EQ(callOp.getNumOperands(), 1u);

  // Switch to indirect
  callOp.setCalleeFromCallable(calleeVal);

  EXPECT_TRUE(isValue(callOp.getCallableForCallee()));
  EXPECT_EQ(callOp.getNumOperands(), 2u);
  EXPECT_EQ(callOp.getOperand(0), calleeVal);
  EXPECT_EQ(callOp.getOperand(1), argVal);
  ArrayAttr argAttrs = callOp.getArgAttrsAttr();
  ASSERT_TRUE(argAttrs);
  ASSERT_EQ(argAttrs.size(), 2u);
  // First entry is empty dict for callee.
  EXPECT_TRUE(llvm::cast<DictionaryAttr>(argAttrs[0]).empty());
  // Second entry preserves the argument's attribute.
  auto argDict = llvm::cast<DictionaryAttr>(argAttrs[1]);
  EXPECT_EQ(argDict.get("test.attr"), StringAttr::get(&context, "arg0"));
}

TEST_F(
    FIRCallInterfaceTest, setCalleeFromCallable_indirectToDirect_withArgAttrs) {
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto i32Ty = builder.getI32Type();
  auto funcType = builder.getFunctionType({i32Ty}, {});
  auto containerType = builder.getFunctionType({funcType, i32Ty}, {});
  auto [func, block] =
      createModuleWithFunction(builder, loc, "container", containerType);
  Value calleeVal = block->getArgument(0);
  Value argVal = block->getArgument(1);

  // Indirect call with callee + one argument
  auto callOp = fir::CallOp::create(builder, loc, SymbolRefAttr{},
      llvm::ArrayRef<mlir::Type>{}, ValueRange{calleeVal, argVal});
  callOp->setAttr(callOp.getArgAttrsAttrName(),
      makeArgAttrs(&context,
          {DictionaryAttr::get(&context, {}),
              makeTestArgDict(&context, "arg0")}));
  ASSERT_TRUE(isValue(callOp.getCallableForCallee()));
  EXPECT_EQ(callOp.getNumOperands(), 2u);

  // Switch to direct
  auto callTargetRef = FlatSymbolRefAttr::get(&context, "direct_target");
  callOp.setCalleeFromCallable(callTargetRef);

  EXPECT_TRUE(isSymbolRef(callOp.getCallableForCallee()));
  EXPECT_EQ(callOp.getNumOperands(), 1u);
  EXPECT_EQ(callOp.getOperand(0), argVal);
  ArrayAttr argAttrs = callOp.getArgAttrsAttr();
  ASSERT_TRUE(argAttrs);
  ASSERT_EQ(argAttrs.size(), 1u);
  EXPECT_EQ(llvm::cast<DictionaryAttr>(argAttrs[0]).get("test.attr"),
      StringAttr::get(&context, "arg0"));
}

TEST_F(FIRCallInterfaceTest,
    setCalleeFromCallable_indirectToIndirect_withArgAttrs) {
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto i32Ty = builder.getI32Type();
  auto funcType = builder.getFunctionType({i32Ty}, {});
  auto containerType = builder.getFunctionType({funcType, funcType, i32Ty}, {});
  auto [func, block] =
      createModuleWithFunction(builder, loc, "container", containerType);
  Value callee0 = block->getArgument(0);
  Value callee1 = block->getArgument(1);
  Value argVal = block->getArgument(2);

  // Indirect call with one argument and arg_attrs
  auto callOp = fir::CallOp::create(builder, loc, SymbolRefAttr{},
      llvm::ArrayRef<mlir::Type>{}, ValueRange{callee0, argVal});
  callOp->setAttr(callOp.getArgAttrsAttrName(),
      makeArgAttrs(&context,
          {DictionaryAttr::get(&context, {}),
              makeTestArgDict(&context, "arg0")}));
  ASSERT_TRUE(isValue(callOp.getCallableForCallee()));
  EXPECT_EQ(callOp.getNumOperands(), 2u);

  // Switch to other indirect callee
  callOp.setCalleeFromCallable(callee1);

  EXPECT_TRUE(isValue(callOp.getCallableForCallee()));
  EXPECT_EQ(callOp.getNumOperands(), 2u);
  EXPECT_EQ(callOp.getOperand(0), callee1);
  EXPECT_EQ(callOp.getOperand(1), argVal);
  ArrayAttr argAttrs = callOp.getArgAttrsAttr();
  ASSERT_TRUE(argAttrs);
  ASSERT_EQ(argAttrs.size(), 2u);
  EXPECT_TRUE(llvm::cast<DictionaryAttr>(argAttrs[0]).empty());
  EXPECT_EQ(llvm::cast<DictionaryAttr>(argAttrs[1]).get("test.attr"),
      StringAttr::get(&context, "arg0"));
}
