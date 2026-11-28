//===- CallOpTest.cpp - Unit tests for cir.call helpers -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

#include <gtest/gtest.h>

using namespace mlir;

namespace {

class CIRCallOpTest : public ::testing::Test {
protected:
  CIRCallOpTest() { context.loadDialect<cir::CIRDialect>(); }

  OwningOpRef<ModuleOp> parse(StringRef ir) {
    auto module = parseSourceString<ModuleOp>(ir, &context);
    EXPECT_TRUE(module) << "failed to parse IR";
    return module;
  }

  MLIRContext context;
};

// A callee that carries func_info, a callee that carries none, an indirect
// call with no callee symbol at all, and a cir.try_call to the marked callee.
constexpr const char *moduleText = R"CIR(
!s32i = !cir.int<s, 32>
!rec_S = !cir.struct<"S" {data !s32i}>

module {
  cir.func @marked() func_info<#cir.cxx_ctor<!rec_S, default>> {
    cir.return
  }
  cir.func @plain() {
    cir.return
  }
  cir.func @caller(%arg0: !cir.ptr<!cir.func<()>>) {
    cir.call @marked() : () -> ()
    cir.call @plain() : () -> ()
    cir.call %arg0() : (!cir.ptr<!cir.func<()>>) -> ()
    cir.return
  }
  cir.func @try_caller() {
    cir.try_call @marked() ^bb1, ^bb2 : () -> ()
  ^bb1:
    cir.return
  ^bb2:
    cir.return
  }
}
)CIR";

TEST_F(CIRCallOpTest, ResolveCallee) {
  OwningOpRef<ModuleOp> module = parse(moduleText);
  ASSERT_TRUE(module);

  SmallVector<cir::CallOp> calls;
  module->walk([&](cir::CallOp op) { calls.push_back(op); });
  ASSERT_EQ(calls.size(), 3u);

  SymbolTableCollection symbolTable;

  // A direct call resolves to its callee, and the caller can then read the
  // facts recorded on it, such as the func_info attribute.
  cir::FuncOp marked = calls[0].resolveCalleeInTable(symbolTable);
  ASSERT_TRUE(marked);
  auto ctor = dyn_cast_or_null<cir::CXXCtorAttr>(marked.getFuncInfoAttr());
  ASSERT_TRUE(ctor);
  EXPECT_EQ(ctor.getCtorKind(), cir::CtorKind::Default);

  // A callee without func_info still resolves, and the attribute reads null.
  cir::FuncOp plain = calls[1].resolveCalleeInTable(symbolTable);
  ASSERT_TRUE(plain);
  EXPECT_FALSE(plain.getFuncInfoAttr());

  // An indirect call has no callee symbol to resolve.
  EXPECT_FALSE(calls[2].resolveCalleeInTable(symbolTable));

  // cir.try_call resolves through the same implementation.
  SmallVector<cir::TryCallOp> tryCalls;
  module->walk([&](cir::TryCallOp op) { tryCalls.push_back(op); });
  ASSERT_EQ(tryCalls.size(), 1u);
  EXPECT_EQ(tryCalls[0].resolveCalleeInTable(symbolTable), marked);
}

// Functions with the same signature that calls can be retargeted between, and
// one direct and one indirect call that each pass a single argument carrying
// arg_attrs.
constexpr const char *retargetModuleText = R"CIR(
!s32i = !cir.int<s, 32>
!fty = !cir.func<(!s32i)>

module {
  cir.func @target(%a: !s32i) {
    cir.return
  }
  cir.func @other(%a: !s32i) {
    cir.return
  }
  cir.func @caller(%fp0: !cir.ptr<!fty>, %fp1: !cir.ptr<!fty>, %x: !s32i) {
    cir.call @target(%x) : (!s32i {cir.test = "arg0"}) -> ()
    cir.call %fp0(%x) : (!cir.ptr<!fty>, !s32i {cir.test = "arg0"}) -> ()
    cir.return
  }
  cir.func @try_caller(%fp0: !cir.ptr<!fty>, %x: !s32i) {
    cir.try_call @target(%x) ^bb1, ^bb2 : (!s32i) -> ()
  ^bb1:
    cir.return
  ^bb2:
    cir.return
  }
}
)CIR";

class CIRCallOpRetargetTest : public CIRCallOpTest {
protected:
  void SetUp() override {
    module = parse(retargetModuleText);
    ASSERT_TRUE(module);
    module->walk([&](cir::CallOp op) { calls.push_back(op); });
    ASSERT_EQ(calls.size(), 2u);
    auto caller = module->lookupSymbol<cir::FuncOp>("caller");
    ASSERT_TRUE(caller);
    fp0 = caller.getArgument(0);
    fp1 = caller.getArgument(1);
    x = caller.getArgument(2);
  }

  cir::CallOp directCall() { return calls[0]; }
  cir::CallOp indirectCall() { return calls[1]; }

  // The call still passes %x as its only argument, with its arg_attrs intact.
  static void expectArgPreserved(cir::CallOp call, Value x) {
    ASSERT_EQ(call.getNumArgOperands(), 1u);
    EXPECT_EQ(call.getArgOperand(0), x);
    ArrayAttr argAttrs = call.getArgAttrsAttr();
    ASSERT_TRUE(argAttrs);
    ASSERT_EQ(argAttrs.size(), 1u);
    EXPECT_EQ(cast<DictionaryAttr>(argAttrs[0]).get("cir.test"),
              StringAttr::get(call.getContext(), "arg0"));
  }

  static StringRef calleeName(CallOpInterface call) {
    return cast<SymbolRefAttr>(call.getCallableForCallee())
        .getRootReference()
        .getValue();
  }

  OwningOpRef<ModuleOp> module;
  SmallVector<cir::CallOp> calls;
  Value fp0, fp1, x;
};

TEST_F(CIRCallOpRetargetTest, DirectToDirect) {
  cir::CallOp call = directCall();
  call.setCalleeFromCallable(FlatSymbolRefAttr::get(&context, "other"));

  EXPECT_FALSE(call.isIndirect());
  EXPECT_EQ(calleeName(call), "other");
  EXPECT_EQ(call->getNumOperands(), 1u);
  expectArgPreserved(call, x);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST_F(CIRCallOpRetargetTest, DirectToIndirect) {
  cir::CallOp call = directCall();
  call.setCalleeFromCallable(fp0);

  EXPECT_TRUE(call.isIndirect());
  EXPECT_FALSE(call.getCalleeAttr());
  EXPECT_EQ(cast<Value>(call.getCallableForCallee()), fp0);
  ASSERT_EQ(call->getNumOperands(), 2u);
  EXPECT_EQ(call->getOperand(0), fp0);
  expectArgPreserved(call, x);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST_F(CIRCallOpRetargetTest, IndirectToDirect) {
  cir::CallOp call = indirectCall();
  call.setCalleeFromCallable(FlatSymbolRefAttr::get(&context, "target"));

  EXPECT_FALSE(call.isIndirect());
  EXPECT_EQ(calleeName(call), "target");
  ASSERT_EQ(call->getNumOperands(), 1u);
  EXPECT_EQ(call->getOperand(0), x);
  expectArgPreserved(call, x);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST_F(CIRCallOpRetargetTest, IndirectToIndirect) {
  cir::CallOp call = indirectCall();
  call.setCalleeFromCallable(fp1);

  EXPECT_TRUE(call.isIndirect());
  EXPECT_EQ(cast<Value>(call.getCallableForCallee()), fp1);
  ASSERT_EQ(call->getNumOperands(), 2u);
  EXPECT_EQ(call->getOperand(0), fp1);
  expectArgPreserved(call, x);
  EXPECT_TRUE(succeeded(verify(*module)));
}

// cir.try_call shares the implementation; its successors must be untouched.
TEST_F(CIRCallOpRetargetTest, TryCallRoundTrip) {
  SmallVector<cir::TryCallOp> tryCalls;
  module->walk([&](cir::TryCallOp op) { tryCalls.push_back(op); });
  ASSERT_EQ(tryCalls.size(), 1u);
  cir::TryCallOp call = tryCalls[0];
  Value tryFp = call->getParentOfType<cir::FuncOp>().getArgument(0);
  Value tryX = call->getParentOfType<cir::FuncOp>().getArgument(1);
  Block *normal = call.getNormalDest();
  Block *unwind = call.getUnwindDest();

  call.setCalleeFromCallable(tryFp);
  EXPECT_TRUE(call.isIndirect());
  ASSERT_EQ(call->getNumOperands(), 2u);
  EXPECT_EQ(call->getOperand(0), tryFp);
  EXPECT_EQ(call->getOperand(1), tryX);
  EXPECT_TRUE(succeeded(verify(*module)));

  call.setCalleeFromCallable(FlatSymbolRefAttr::get(&context, "other"));
  EXPECT_FALSE(call.isIndirect());
  EXPECT_EQ(calleeName(call), "other");
  ASSERT_EQ(call->getNumOperands(), 1u);
  EXPECT_EQ(call->getOperand(0), tryX);
  EXPECT_EQ(call.getNormalDest(), normal);
  EXPECT_EQ(call.getUnwindDest(), unwind);
  EXPECT_TRUE(succeeded(verify(*module)));
}

} // namespace
