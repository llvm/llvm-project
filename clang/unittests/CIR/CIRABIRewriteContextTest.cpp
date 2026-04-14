//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for CIRABIRewriteContext, the CIR dialect's concrete
// implementation of the shared ABIRewriteContext interface.  Each test
// constructs a FunctionClassification manually (no ABI library needed)
// and verifies the resulting IR after rewriting.
//
//===----------------------------------------------------------------------===//

#include "mlir/ABI/ABIRewriteContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "gtest/gtest.h"

// The header is private to the Transforms library, so we include it
// via the path relative to the source tree.  The CMakeLists arranges
// the include directories.
#include "../../lib/CIR/Dialect/Transforms/TargetLowering/CIRABIRewriteContext.h"

using namespace mlir;
using namespace mlir::abi;

namespace {

class CIRABIRewriteTest : public ::testing::Test {
protected:
  CIRABIRewriteTest() : builder(&context), loc(UnknownLoc::get(&context)) {
    context.loadDialect<cir::CIRDialect>();
  }

  MLIRContext context;
  OpBuilder builder;
  Location loc;

  /// Create a ModuleOp containing a single CIR FuncOp with the given
  /// argument types and return type.  If \p addBody is true, the
  /// function gets an entry block with a cir.return (returning its
  /// first result-typed block arg if non-void, or void otherwise).
  std::pair<ModuleOp, cir::FuncOp> createFunc(StringRef name,
                                              ArrayRef<Type> argTypes,
                                              Type retType,
                                              bool addBody = true) {
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());

    auto funcTy = cir::FuncType::get(argTypes, retType);
    auto funcOp = cir::FuncOp::create(builder, loc, name, funcTy);

    if (addBody) {
      Block *entry = funcOp.addEntryBlock();
      builder.setInsertionPointToEnd(entry);
      if (isa<cir::VoidType>(retType))
        cir::ReturnOp::create(builder, loc);
      else
        cir::ReturnOp::create(builder, loc,
                              mlir::ValueRange{entry->getArgument(0)});
    }

    return {module, funcOp};
  }

  /// Create a ModuleOp containing a caller function that calls a
  /// callee.  The caller passes its own block arguments to the callee.
  struct CallFixture {
    ModuleOp module;
    cir::FuncOp callee;
    cir::FuncOp caller;
    cir::CallOp callOp;
  };

  CallFixture createCallPair(StringRef calleeName, ArrayRef<Type> argTypes,
                             Type retType) {
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());

    auto funcTy = cir::FuncType::get(argTypes, retType);

    // Callee (declaration only).
    auto callee = cir::FuncOp::create(builder, loc, calleeName, funcTy);

    // Caller with a body that calls the callee.
    auto caller = cir::FuncOp::create(builder, loc, "caller", funcTy);
    Block *entry = caller.addEntryBlock();
    builder.setInsertionPointToEnd(entry);

    SmallVector<Value> args;
    for (unsigned i = 0; i < argTypes.size(); ++i)
      args.push_back(entry->getArgument(i));

    cir::CallOp call;
    if (isa<cir::VoidType>(retType)) {
      auto voidTy = cir::VoidType::get(&context);
      call = cir::CallOp::create(
          builder, loc, mlir::FlatSymbolRefAttr::get(&context, calleeName),
          voidTy, args);
      cir::ReturnOp::create(builder, loc);
    } else {
      call = cir::CallOp::create(
          builder, loc, mlir::FlatSymbolRefAttr::get(&context, calleeName),
          retType, args);
      cir::ReturnOp::create(builder, loc, mlir::ValueRange{call.getResult()});
    }

    return {module, callee, caller, call};
  }
};

// ---- rewriteFunctionDefinition tests ----

TEST_F(CIRABIRewriteTest, DirectPassthrough) {
  auto i32Ty = cir::IntType::get(&context, 32, true);
  auto [module, funcOp] = createFunc("f", {i32Ty}, i32Ty);

  FunctionClassification fc;
  fc.returnInfo = ArgClassification::getDirect();
  fc.argInfos.push_back(ArgClassification::getDirect());

  cir::CIRABIRewriteContext rewriteCtx(module);
  OpBuilder rewriter(funcOp);
  ASSERT_TRUE(
      succeeded(rewriteCtx.rewriteFunctionDefinition(funcOp, fc, rewriter)));

  auto fnTy = cast<cir::FuncType>(funcOp.getFunctionType());
  EXPECT_EQ(fnTy.getInputs().size(), 1u);
  EXPECT_EQ(fnTy.getInputs()[0], i32Ty);
  EXPECT_EQ(fnTy.getReturnType(), i32Ty);

  module->erase();
}

TEST_F(CIRABIRewriteTest, DirectReturnCoercion) {
  auto i32Ty = cir::IntType::get(&context, 32, true);
  auto i64Ty = cir::IntType::get(&context, 64, false);
  auto [module, funcOp] = createFunc("f", {i32Ty}, i32Ty);

  FunctionClassification fc;
  fc.returnInfo = ArgClassification::getDirect(i64Ty);
  fc.argInfos.push_back(ArgClassification::getDirect());

  cir::CIRABIRewriteContext rewriteCtx(module);
  OpBuilder rewriter(funcOp);
  ASSERT_TRUE(
      succeeded(rewriteCtx.rewriteFunctionDefinition(funcOp, fc, rewriter)));

  auto fnTy = cast<cir::FuncType>(funcOp.getFunctionType());
  EXPECT_EQ(fnTy.getReturnType(), i64Ty);

  module->erase();
}

TEST_F(CIRABIRewriteTest, ExtendArg) {
  auto i8Ty = cir::IntType::get(&context, 8, true);
  auto i32Ty = cir::IntType::get(&context, 32, true);
  auto voidTy = cir::VoidType::get(&context);
  auto [module, funcOp] = createFunc("f", {i8Ty}, voidTy);

  FunctionClassification fc;
  fc.returnInfo = ArgClassification::getDirect();
  fc.argInfos.push_back(ArgClassification::getExtend(i32Ty, true));

  cir::CIRABIRewriteContext rewriteCtx(module);
  OpBuilder rewriter(funcOp);
  ASSERT_TRUE(
      succeeded(rewriteCtx.rewriteFunctionDefinition(funcOp, fc, rewriter)));

  auto fnTy = cast<cir::FuncType>(funcOp.getFunctionType());
  EXPECT_EQ(fnTy.getInputs().size(), 1u);
  EXPECT_EQ(fnTy.getInputs()[0], i32Ty);

  // Verify signext attribute was attached.
  auto argAttrs = funcOp->getAttrOfType<ArrayAttr>("arg_attrs");
  ASSERT_TRUE(argAttrs != nullptr);
  ASSERT_EQ(argAttrs.size(), 1u);
  auto dict = cast<DictionaryAttr>(argAttrs[0]);
  EXPECT_TRUE(dict.get("llvm.signext") != nullptr);

  // Verify the entry block has a cir.cast (integral) to adapt i32
  // back to i8 for body uses.
  Block &entry = funcOp->getRegion(0).front();
  bool foundCast = false;
  for (Operation &op : entry) {
    if (auto cast = dyn_cast<cir::CastOp>(op)) {
      if (cast.getKind() == cir::CastKind::integral) {
        EXPECT_EQ(cast.getResult().getType(), i8Ty);
        foundCast = true;
      }
    }
  }
  EXPECT_TRUE(foundCast);

  module->erase();
}

TEST_F(CIRABIRewriteTest, ExtendReturn) {
  auto i8Ty = cir::IntType::get(&context, 8, true);
  auto i32Ty = cir::IntType::get(&context, 32, true);
  auto [module, funcOp] = createFunc("f", {i8Ty}, i8Ty);

  FunctionClassification fc;
  fc.returnInfo = ArgClassification::getExtend(i32Ty, false);
  fc.argInfos.push_back(ArgClassification::getDirect());

  cir::CIRABIRewriteContext rewriteCtx(module);
  OpBuilder rewriter(funcOp);
  ASSERT_TRUE(
      succeeded(rewriteCtx.rewriteFunctionDefinition(funcOp, fc, rewriter)));

  auto fnTy = cast<cir::FuncType>(funcOp.getFunctionType());
  EXPECT_EQ(fnTy.getReturnType(), i32Ty);

  // Verify zeroext attribute on return.
  auto resAttrs = funcOp->getAttrOfType<ArrayAttr>("res_attrs");
  ASSERT_TRUE(resAttrs != nullptr);
  ASSERT_EQ(resAttrs.size(), 1u);
  auto dict = cast<DictionaryAttr>(resAttrs[0]);
  EXPECT_TRUE(dict.get("llvm.zeroext") != nullptr);

  module->erase();
}

TEST_F(CIRABIRewriteTest, IgnoreReturn) {
  auto i32Ty = cir::IntType::get(&context, 32, true);
  auto [module, funcOp] = createFunc("f", {i32Ty}, i32Ty);

  FunctionClassification fc;
  fc.returnInfo = ArgClassification::getIgnore();
  fc.argInfos.push_back(ArgClassification::getDirect());

  cir::CIRABIRewriteContext rewriteCtx(module);
  OpBuilder rewriter(funcOp);
  ASSERT_TRUE(
      succeeded(rewriteCtx.rewriteFunctionDefinition(funcOp, fc, rewriter)));

  auto fnTy = cast<cir::FuncType>(funcOp.getFunctionType());
  auto voidTy = cir::VoidType::get(&context);
  EXPECT_EQ(fnTy.getReturnType(), voidTy);

  module->erase();
}

TEST_F(CIRABIRewriteTest, IgnoreArg) {
  auto i32Ty = cir::IntType::get(&context, 32, true);
  auto voidTy = cir::VoidType::get(&context);
  auto [module, funcOp] = createFunc("f", {i32Ty}, voidTy);

  FunctionClassification fc;
  fc.returnInfo = ArgClassification::getDirect();
  fc.argInfos.push_back(ArgClassification::getIgnore());

  cir::CIRABIRewriteContext rewriteCtx(module);
  OpBuilder rewriter(funcOp);
  ASSERT_TRUE(
      succeeded(rewriteCtx.rewriteFunctionDefinition(funcOp, fc, rewriter)));

  auto fnTy = cast<cir::FuncType>(funcOp.getFunctionType());
  EXPECT_EQ(fnTy.getInputs().size(), 0u);

  Block &entry = funcOp->getRegion(0).front();
  EXPECT_EQ(entry.getNumArguments(), 0u);

  module->erase();
}

TEST_F(CIRABIRewriteTest, DeclarationRewrite) {
  auto i8Ty = cir::IntType::get(&context, 8, true);
  auto i32Ty = cir::IntType::get(&context, 32, true);
  auto [module, funcOp] = createFunc("f", {i8Ty}, i8Ty, /*addBody=*/false);

  FunctionClassification fc;
  fc.returnInfo = ArgClassification::getExtend(i32Ty, true);
  fc.argInfos.push_back(ArgClassification::getExtend(i32Ty, true));

  cir::CIRABIRewriteContext rewriteCtx(module);
  OpBuilder rewriter(funcOp);
  ASSERT_TRUE(
      succeeded(rewriteCtx.rewriteFunctionDefinition(funcOp, fc, rewriter)));

  auto fnTy = cast<cir::FuncType>(funcOp.getFunctionType());
  EXPECT_EQ(fnTy.getInputs()[0], i32Ty);
  EXPECT_EQ(fnTy.getReturnType(), i32Ty);

  // Verify both signext attributes.
  auto argAttrs = funcOp->getAttrOfType<ArrayAttr>("arg_attrs");
  ASSERT_TRUE(argAttrs != nullptr);
  auto dict = cast<DictionaryAttr>(argAttrs[0]);
  EXPECT_TRUE(dict.get("llvm.signext") != nullptr);

  auto resAttrs = funcOp->getAttrOfType<ArrayAttr>("res_attrs");
  ASSERT_TRUE(resAttrs != nullptr);
  auto rdict = cast<DictionaryAttr>(resAttrs[0]);
  EXPECT_TRUE(rdict.get("llvm.signext") != nullptr);

  module->erase();
}

// ---- rewriteCallSite tests ----

TEST_F(CIRABIRewriteTest, CallSiteDirectPassthrough) {
  auto i32Ty = cir::IntType::get(&context, 32, true);
  auto fixture = createCallPair("callee", {i32Ty}, i32Ty);

  FunctionClassification fc;
  fc.returnInfo = ArgClassification::getDirect();
  fc.argInfos.push_back(ArgClassification::getDirect());

  cir::CIRABIRewriteContext rewriteCtx(fixture.module);
  OpBuilder rewriter(fixture.callOp);
  ASSERT_TRUE(
      succeeded(rewriteCtx.rewriteCallSite(fixture.callOp, fc, rewriter)));

  // The original call should still be there (no changes needed).
  EXPECT_EQ(fixture.callOp->getNumResults(), 1u);
  EXPECT_EQ(fixture.callOp->getResult(0).getType(), i32Ty);

  fixture.module->erase();
}

TEST_F(CIRABIRewriteTest, CallSiteExtendArg) {
  auto i8Ty = cir::IntType::get(&context, 8, true);
  auto i32Ty = cir::IntType::get(&context, 32, true);
  auto voidTy = cir::VoidType::get(&context);
  auto fixture = createCallPair("callee", {i8Ty}, voidTy);

  FunctionClassification fc;
  fc.returnInfo = ArgClassification::getDirect();
  fc.argInfos.push_back(ArgClassification::getExtend(i32Ty, true));

  cir::CIRABIRewriteContext rewriteCtx(fixture.module);
  OpBuilder rewriter(fixture.callOp);
  ASSERT_TRUE(
      succeeded(rewriteCtx.rewriteCallSite(fixture.callOp, fc, rewriter)));

  // The old call was erased and replaced.  Look for a CallOp whose
  // argument is i32 (the extended type).
  Block &callerEntry = fixture.caller->getRegion(0).front();
  cir::CallOp newCall;
  for (Operation &op : callerEntry)
    if (auto c = dyn_cast<cir::CallOp>(op))
      newCall = c;
  ASSERT_TRUE(newCall != nullptr);
  EXPECT_EQ(newCall.getArgOperands()[0].getType(), i32Ty);

  fixture.module->erase();
}

TEST_F(CIRABIRewriteTest, CallSiteIgnoreReturn) {
  auto i32Ty = cir::IntType::get(&context, 32, true);
  auto fixture = createCallPair("callee", {i32Ty}, i32Ty);

  FunctionClassification fc;
  fc.returnInfo = ArgClassification::getIgnore();
  fc.argInfos.push_back(ArgClassification::getDirect());

  cir::CIRABIRewriteContext rewriteCtx(fixture.module);
  OpBuilder rewriter(fixture.callOp);
  ASSERT_TRUE(
      succeeded(rewriteCtx.rewriteCallSite(fixture.callOp, fc, rewriter)));

  // Find the replacement void call.
  Block &callerEntry = fixture.caller->getRegion(0).front();
  cir::CallOp newCall;
  for (Operation &op : callerEntry)
    if (auto c = dyn_cast<cir::CallOp>(op))
      newCall = c;
  ASSERT_TRUE(newCall != nullptr);
  EXPECT_EQ(newCall.getNumResults(), 0u);

  fixture.module->erase();
}

TEST_F(CIRABIRewriteTest, CallSiteIgnoreArg) {
  auto i32Ty = cir::IntType::get(&context, 32, true);
  auto voidTy = cir::VoidType::get(&context);
  auto fixture = createCallPair("callee", {i32Ty}, voidTy);

  FunctionClassification fc;
  fc.returnInfo = ArgClassification::getDirect();
  fc.argInfos.push_back(ArgClassification::getIgnore());

  cir::CIRABIRewriteContext rewriteCtx(fixture.module);
  OpBuilder rewriter(fixture.callOp);
  ASSERT_TRUE(
      succeeded(rewriteCtx.rewriteCallSite(fixture.callOp, fc, rewriter)));

  // Find the replacement call -- it should have zero args.
  Block &callerEntry = fixture.caller->getRegion(0).front();
  cir::CallOp newCall;
  for (Operation &op : callerEntry)
    if (auto c = dyn_cast<cir::CallOp>(op))
      newCall = c;
  ASSERT_TRUE(newCall != nullptr);
  EXPECT_EQ(newCall.getArgOperands().size(), 0u);

  fixture.module->erase();
}

} // namespace
