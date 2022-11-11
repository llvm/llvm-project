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

#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Basic/LangStandard.h"
#include "clang/CodeGen/BackendUtil.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/Support/FormatVariadic.h"
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
      FrontendInputFile("test.cc", Language::CXX));
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

TEST(CodeGenTest, CodeGenFromIRMemBuffer) {
  auto Invocation = std::make_shared<CompilerInvocation>();
  std::unique_ptr<MemoryBuffer> MemBuffer =
      MemoryBuffer::getMemBuffer("", "test.ll");
  Invocation->getFrontendOpts().Inputs.push_back(
      FrontendInputFile(*MemBuffer, Language::LLVM_IR));
  Invocation->getFrontendOpts().ProgramAction = frontend::EmitLLVMOnly;
  Invocation->getTargetOpts().Triple = "i386-unknown-linux-gnu";
  CompilerInstance Compiler;
  Compiler.setInvocation(std::move(Invocation));
  Compiler.createDiagnostics();
  EXPECT_TRUE(Compiler.hasDiagnostics());

  EmitLLVMOnlyAction Action;
  bool Success = Compiler.ExecuteAction(Action);
  EXPECT_TRUE(Success);
}

TEST(CodeGenTest, DebugInfoCWDCodeGen) {
  // Check that debug info is accessing the current working directory from the
  // VFS instead of calling \p llvm::sys::fs::current_path() directly.

  auto VFS = std::make_unique<llvm::vfs::InMemoryFileSystem>();
  VFS->setCurrentWorkingDirectory("/in-memory-fs-cwd");
  auto Sept = llvm::sys::path::get_separator();
  std::string TestPath =
      std::string(llvm::formatv("{0}in-memory-fs-cwd{0}test.cpp", Sept));
  VFS->addFile(TestPath, 0, llvm::MemoryBuffer::getMemBuffer("int x;\n"));

  auto Invocation = std::make_shared<CompilerInvocation>();
  Invocation->getFrontendOpts().Inputs.push_back(
      FrontendInputFile("test.cpp", Language::CXX));
  Invocation->getFrontendOpts().ProgramAction = EmitLLVM;
  Invocation->getTargetOpts().Triple = "x86_64-unknown-linux-gnu";
  Invocation->getCodeGenOpts().setDebugInfo(codegenoptions::FullDebugInfo);
  CompilerInstance Compiler;

  SmallString<256> IRBuffer;
  Compiler.setOutputStream(std::make_unique<raw_svector_ostream>(IRBuffer));
  Compiler.setInvocation(std::move(Invocation));
  Compiler.createDiagnostics();
  Compiler.createFileManager(std::move(VFS));

  EmitLLVMAction Action;
  bool Success = Compiler.ExecuteAction(Action);
  EXPECT_TRUE(Success);

  SmallString<128> RealCWD;
  llvm::sys::fs::current_path(RealCWD);
  EXPECT_TRUE(!RealCWD.empty());
  EXPECT_FALSE(IRBuffer.str().contains(RealCWD));
  EXPECT_TRUE(IRBuffer.str().contains("in-memory-fs-cwd"));
}
}
