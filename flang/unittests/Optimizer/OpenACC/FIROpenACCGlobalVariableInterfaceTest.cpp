//===---------- FIROpenACCGlobalVariableInterfaceTest.cpp -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/OpenACC/Support/RegisterOpenACCExtensions.h"
#include "flang/Optimizer/Support/InitFIR.h"

using namespace mlir;

namespace {

struct FIROpenACCGlobalVariableInterfaceTest : public testing::Test {
  void SetUp() override {
    DialectRegistry registry;
    fir::acc::registerOpenACCExtensions(registry);
    context.appendDialectRegistry(registry);
    fir::support::loadDialects(context);
    module = ModuleOp::create(UnknownLoc::get(&context));
  }

  fir::GlobalOp createGlobal(StringRef name, fir::LinkageAttr linkage) {
    OpBuilder builder(module->getBodyRegion());
    return fir::GlobalOp::create(builder, module->getLoc(), name,
        /*isConstant=*/false, /*isTarget=*/false, builder.getF32Type(),
        Attribute(), linkage);
  }

  acc::GlobalVariableOpInterface getInterface(fir::GlobalOp global) {
    return dyn_cast<acc::GlobalVariableOpInterface>(global.getOperation());
  }

  MLIRContext context;
  OwningOpRef<ModuleOp> module;
};

TEST_F(FIROpenACCGlobalVariableInterfaceTest, InternalLinkage) {
  // fir.global internal @internal_global : f32
  // FIR has no private linkage, so internal linkage is local linkage.
  fir::GlobalOp global = createGlobal("internal_global",
      fir::LinkageAttr::get(&context, fir::LinkageEnum::Internal));
  acc::GlobalVariableOpInterface iface = getInterface(global);
  ASSERT_TRUE(iface);
  EXPECT_TRUE(iface.hasLocalLinkage());
}

TEST_F(FIROpenACCGlobalVariableInterfaceTest, ExternalLinkage) {
  // fir.global external @external_global : f32
  // hasLocalLinkage() is false for explicit external linkage.
  fir::GlobalOp global = createGlobal("external_global",
      fir::LinkageAttr::get(&context, fir::LinkageEnum::External));
  acc::GlobalVariableOpInterface iface = getInterface(global);
  ASSERT_TRUE(iface);
  EXPECT_FALSE(iface.hasLocalLinkage());
}

TEST_F(FIROpenACCGlobalVariableInterfaceTest, NoLinkage) {
  // fir.global @no_linkage_global : f32
  // hasLocalLinkage() is false when the linkage attribute is absent.
  fir::GlobalOp global = createGlobal("no_linkage_global", fir::LinkageAttr());
  acc::GlobalVariableOpInterface iface = getInterface(global);
  ASSERT_TRUE(iface);
  EXPECT_FALSE(iface.hasLocalLinkage());
}

} // namespace
