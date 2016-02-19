//===- unittests/Frontend/CodeGenActionTest.cpp --- FrontendAction tests --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Unit tests for CodeGenAction.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/CompilerInstance.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/CodeGen/BackendUtil.h"
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
  CompilerInvocation *Invocation = new CompilerInvocation;
  Invocation->getPreprocessorOpts().addRemappedFile(
      "test.cc",
      MemoryBuffer::getMemBuffer("").release());
  Invocation->getFrontendOpts().Inputs.push_back(
      FrontendInputFile("test.cc", IK_CXX));
  Invocation->getFrontendOpts().ProgramAction = EmitLLVM;
  Invocation->getTargetOpts().Triple = "i386-unknown-linux-gnu";
  CompilerInstance Compiler;
  Compiler.setInvocation(Invocation);
  Compiler.createDiagnostics();
  EXPECT_TRUE(Compiler.hasDiagnostics());

  std::unique_ptr<FrontendAction> Act(new NullCodeGenAction);
  bool Success = Compiler.ExecuteAction(*Act);
  EXPECT_TRUE(Success);
}

}
