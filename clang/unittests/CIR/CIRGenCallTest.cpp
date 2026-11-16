//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../../lib/CIR/CodeGen/CIRGenCall.h"
#include "../../lib/CIR/CodeGen/CIRGenModule.h"
#include "../../lib/CIR/CodeGen/TargetInfo.h"
#include "mlir/IR/MLIRContext.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/CodeGenOptions.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"
#include <memory>

using namespace clang;
using namespace clang::CIRGen;

namespace {

class NoProtoTargetInfo : public TargetCIRGenInfo {
  bool useVariadicConvention;

public:
  NoProtoTargetInfo(CIRGenTypes &types, bool useVariadicConvention)
      : TargetCIRGenInfo(std::make_unique<ABIInfo>(types)),
        useVariadicConvention(useVariadicConvention) {}

  bool isNoProtoCallVariadic(const FunctionNoProtoType *) const override {
    return useVariadicConvention;
  }
};

class CIRGenCallTest : public ::testing::Test {
protected:
  void SetUp() override {
    ast = tooling::buildASTFromCodeWithArgs(
        "int callee();", {"-std=c11", "-target", "x86_64-unknown-linux-gnu"},
        "test.c");
    ASSERT_TRUE(ast);
    mlirContext.loadDialect<cir::CIRDialect>();
    cgm = std::make_unique<CIRGenModule>(mlirContext, ast->getASTContext(),
                                         codeGenOptions, ast->getDiagnostics());
  }

  const FunctionNoProtoType *getFunctionType() {
    ASTContext &context = ast->getASTContext();
    return context.getFunctionNoProtoType(context.IntTy)
        ->castAs<FunctionNoProtoType>();
  }

  std::unique_ptr<ASTUnit> ast;
  CodeGenOptions codeGenOptions;
  mlir::MLIRContext mlirContext;
  std::unique_ptr<CIRGenModule> cgm;
};

TEST_F(CIRGenCallTest, NoProtoCallWithNoArguments) {
  CIRGenTypes &types = cgm->getTypes();
  NoProtoTargetInfo variadicTarget(types, true);
  NoProtoTargetInfo fixedTarget(types, false);
  CallArgList args;

  const CIRGenFunctionInfo &variadic =
      types.arrangeFreeFunctionCall(args, getFunctionType(), variadicTarget);
  EXPECT_TRUE(variadic.isVariadic());
  EXPECT_EQ(variadic.getNumRequiredArgs(), 0u);
  cir::FuncType variadicType = types.getFunctionType(variadic);
  EXPECT_TRUE(variadicType.isVarArg());
  EXPECT_EQ(variadicType.getNumInputs(), 0u);

  const CIRGenFunctionInfo &fixed =
      types.arrangeFreeFunctionCall(args, getFunctionType(), fixedTarget);
  EXPECT_FALSE(fixed.isVariadic());
  EXPECT_EQ(fixed.getNumRequiredArgs(), 0u);
  EXPECT_FALSE(types.getFunctionType(fixed).isVarArg());
  EXPECT_NE(&variadic, &fixed);
}

TEST_F(CIRGenCallTest, NoProtoCallRequiresAllProvidedArguments) {
  ASTContext &context = ast->getASTContext();
  CIRGenTypes &types = cgm->getTypes();
  NoProtoTargetInfo target(types, true);
  CallArgList args;
  // Call argument types have already undergone default argument promotions.
  args.add(RValue::getIgnored(), context.IntTy);
  args.add(RValue::getIgnored(), context.IntTy);
  args.add(RValue::getIgnored(), context.DoubleTy);

  const CIRGenFunctionInfo &info =
      types.arrangeFreeFunctionCall(args, getFunctionType(), target);
  EXPECT_TRUE(info.isVariadic());
  EXPECT_EQ(info.getNumRequiredArgs(), 3u);
  EXPECT_EQ(info.argTypeSize(), 3u);

  cir::FuncType functionType = types.getFunctionType(info);
  EXPECT_TRUE(functionType.isVarArg());
  ASSERT_EQ(functionType.getNumInputs(), 3u);
  mlir::Type intType = cir::IntType::get(&mlirContext, 32, true);
  EXPECT_EQ(functionType.getInput(0), intType);
  EXPECT_EQ(functionType.getInput(1), intType);
  EXPECT_EQ(functionType.getInput(2), cir::DoubleType::get(&mlirContext));
}

} // namespace
