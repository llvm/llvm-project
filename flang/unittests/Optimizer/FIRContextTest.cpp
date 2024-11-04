//===- FIRContextTest.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "gtest/gtest.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "llvm/TargetParser/Host.h"
#include <string>

using namespace fir;

struct StringAttributesTests : public testing::Test {
public:
  void SetUp() {
    context.loadDialect<mlir::LLVM::LLVMDialect>();
    kindMap = new KindMapping(&context, kindMapInit, "r42a10c14d28i40l41");
    mod = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  }

  void TearDown() { delete kindMap; }

  mlir::MLIRContext context;
  KindMapping *kindMap{};
  std::string kindMapInit =
      "i10:80,l3:24,a1:8,r54:Double,r62:X86_FP80,r11:PPC_FP128";
  std::string target = "powerpc64le-unknown-linux-gnu";
  std::string targetCPU = "gfx90a";
  std::string targetFeatures = "+gfx9-insts,+wavefrontsize64";
  mlir::ModuleOp mod;
};

TEST_F(StringAttributesTests, moduleStringAttrTest) {
  setTargetTriple(mod, target);
  setKindMapping(mod, *kindMap);
  setTargetCPU(mod, targetCPU);
  setTargetFeatures(mod, targetFeatures);

  auto triple = getTargetTriple(mod);
  EXPECT_EQ(triple.getArch(), llvm::Triple::ArchType::ppc64le);
  EXPECT_EQ(triple.getOS(), llvm::Triple::OSType::Linux);

  auto map = getKindMapping(mod);
  EXPECT_EQ(map.defaultsToString(), "a10c14d28i40l41r42");

  auto mapStr = map.mapToString();
  EXPECT_EQ(mapStr.size(), kindMapInit.size());
  EXPECT_TRUE(mapStr.find("a1:8") != std::string::npos);
  EXPECT_TRUE(mapStr.find("l3:24") != std::string::npos);
  EXPECT_TRUE(mapStr.find("i10:80") != std::string::npos);
  EXPECT_TRUE(mapStr.find("r11:PPC_FP128") != std::string::npos);
  EXPECT_TRUE(mapStr.find("r54:Double") != std::string::npos);
  EXPECT_TRUE(mapStr.find("r62:X86_FP80") != std::string::npos);

  EXPECT_EQ(getTargetCPU(mod), targetCPU);

  auto features = getTargetFeatures(mod);
  auto featuresList = features.getFeatures();
  EXPECT_EQ(features.getFeaturesString(), targetFeatures);
  EXPECT_EQ(featuresList.size(), 2u);
  EXPECT_EQ(featuresList[0].str(), "+gfx9-insts");
  EXPECT_EQ(featuresList[1].str(), "+wavefrontsize64");
}

// main() from gtest_main
