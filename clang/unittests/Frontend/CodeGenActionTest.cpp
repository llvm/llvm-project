//===- unittests/Frontend/CodeGenActionTest.cpp --- FrontendAction tests --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for CodeGenAction.
//
//===----------------------------------------------------------------------===//

#include "clang/CodeGen/BackendUtil.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;
using namespace clang::frontend;

namespace {


class NullCodeGenAction : public CodeGenAction {
public:
  NullCodeGenAction(llvm::LLVMContext *_VMContext = nullptr)
    : CodeGenAction(Backend_EmitMCNull, _VMContext) {}

  // The action does not call methods of ATContext.
  void ExecuteAction() override {
    CompilerInstance &CI = getCompilerInstance();
    if (!CI.hasPreprocessor())
      return;
    if (!CI.hasSema())
      CI.createSema(getTranslationUnitKind(), nullptr);
  }
};


TEST(CodeGenTest, TestNullCodeGen) {
  auto Invocation = std::make_shared<CompilerInvocation>();
  Invocation->getPreprocessorOpts().addRemappedFile(
      "test.cc",
      MemoryBuffer::getMemBuffer("").release());
  Invocation->getFrontendOpts().Inputs.push_back(
      FrontendInputFile("test.cc", InputKind::CXX));
  Invocation->getFrontendOpts().ProgramAction = EmitLLVM;
  Invocation->getTargetOpts().Triple = "i386-unknown-linux-gnu";
  CompilerInstance Compiler;
  Compiler.setInvocation(std::move(Invocation));
  Compiler.createDiagnostics();
  EXPECT_TRUE(Compiler.hasDiagnostics());

  std::unique_ptr<FrontendAction> Act(new NullCodeGenAction);
  bool Success = Compiler.ExecuteAction(*Act);
  EXPECT_TRUE(Success);
}

}
