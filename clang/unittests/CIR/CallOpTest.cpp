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
!rec_S = !cir.struct<"S" {!s32i}>

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

} // namespace
