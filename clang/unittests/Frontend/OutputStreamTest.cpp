//===- unittests/Frontend/OutputStreamTest.cpp --- FrontendAction tests --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/CodeGen/BackendUtil.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/FrontendTool/Utils.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;
using namespace clang::frontend;

namespace {

TEST(FrontendOutputTests, TestOutputStream) {
  auto Invocation = std::make_shared<CompilerInvocation>();
  Invocation->getPreprocessorOpts().addRemappedFile(
      "test.cc", MemoryBuffer::getMemBuffer("").release());
  Invocation->getFrontendOpts().Inputs.push_back(
      FrontendInputFile("test.cc", InputKind::CXX));
  Invocation->getFrontendOpts().ProgramAction = EmitBC;
  Invocation->getTargetOpts().Triple = "i386-unknown-linux-gnu";
  CompilerInstance Compiler;

  SmallVector<char, 256> IRBuffer;
  std::unique_ptr<raw_pwrite_stream> IRStream(
      new raw_svector_ostream(IRBuffer));

  Compiler.setOutputStream(std::move(IRStream));
  Compiler.setInvocation(std::move(Invocation));
  Compiler.createDiagnostics();

  bool Success = ExecuteCompilerInvocation(&Compiler);
  EXPECT_TRUE(Success);
  EXPECT_TRUE(!IRBuffer.empty());
  EXPECT_TRUE(StringRef(IRBuffer.data()).startswith("BC"));
}
}
