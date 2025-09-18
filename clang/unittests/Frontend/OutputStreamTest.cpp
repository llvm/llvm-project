//===- unittests/Frontend/OutputStreamTest.cpp --- FrontendAction tests --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/LangStandard.h"
#include "clang/CodeGen/BackendUtil.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/FrontendTool/Utils.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;
using namespace clang::frontend;

namespace {

TEST(FrontendOutputTests, TestOutputStream) {
  llvm::InitializeAllTargetMCs();
  auto Invocation = std::make_shared<CompilerInvocation>();
  Invocation->getPreprocessorOpts().addRemappedFile(
      "test.cc", MemoryBuffer::getMemBuffer("").release());
  Invocation->getFrontendOpts().Inputs.push_back(
      FrontendInputFile("test.cc", Language::CXX));
  Invocation->getFrontendOpts().ProgramAction = EmitBC;
  Invocation->getTargetOpts().Triple = "i386-unknown-linux-gnu";
  CompilerInstance Compiler(std::move(Invocation));

  SmallVector<char, 256> IRBuffer;
  std::unique_ptr<raw_pwrite_stream> IRStream(
      new raw_svector_ostream(IRBuffer));

  Compiler.setOutputStream(std::move(IRStream));
  Compiler.setVirtualFileSystem(llvm::vfs::getRealFileSystem());
  Compiler.createDiagnostics();

  bool Success = ExecuteCompilerInvocation(&Compiler);
  EXPECT_TRUE(Success);
  EXPECT_TRUE(!IRBuffer.empty());
  EXPECT_TRUE(StringRef(IRBuffer.data()).starts_with("BC"));
}

TEST(FrontendOutputTests, TestVerboseOutputStreamShared) {
  llvm::InitializeAllTargetMCs();
  auto Invocation = std::make_shared<CompilerInvocation>();
  Invocation->getPreprocessorOpts().addRemappedFile(
      "test.cc", MemoryBuffer::getMemBuffer("invalid").release());
  Invocation->getFrontendOpts().Inputs.push_back(
      FrontendInputFile("test.cc", Language::CXX));
  Invocation->getFrontendOpts().ProgramAction = EmitBC;
  Invocation->getTargetOpts().Triple = "i386-unknown-linux-gnu";
  CompilerInstance Compiler(std::move(Invocation));

  std::string VerboseBuffer;
  raw_string_ostream VerboseStream(VerboseBuffer);

  Compiler.setOutputStream(std::make_unique<raw_null_ostream>());
  DiagnosticOptions DiagOpts;
  Compiler.setVirtualFileSystem(llvm::vfs::getRealFileSystem());
  Compiler.createDiagnostics(new TextDiagnosticPrinter(llvm::nulls(), DiagOpts),
                             true);
  Compiler.setVerboseOutputStream(VerboseStream);

  bool Success = ExecuteCompilerInvocation(&Compiler);
  EXPECT_FALSE(Success);
  EXPECT_TRUE(!VerboseBuffer.empty());
  EXPECT_TRUE(StringRef(VerboseBuffer.data()).contains("errors generated"));
}

TEST(FrontendOutputTests, TestVerboseOutputStreamOwned) {
  std::string VerboseBuffer;
  bool Success;
  {
    llvm::InitializeAllTargetMCs();
    auto Invocation = std::make_shared<CompilerInvocation>();
    Invocation->getPreprocessorOpts().addRemappedFile(
        "test.cc", MemoryBuffer::getMemBuffer("invalid").release());
    Invocation->getFrontendOpts().Inputs.push_back(
        FrontendInputFile("test.cc", Language::CXX));
    Invocation->getFrontendOpts().ProgramAction = EmitBC;
    Invocation->getTargetOpts().Triple = "i386-unknown-linux-gnu";
    CompilerInstance Compiler(std::move(Invocation));

    std::unique_ptr<raw_ostream> VerboseStream =
        std::make_unique<raw_string_ostream>(VerboseBuffer);

    Compiler.setOutputStream(std::make_unique<raw_null_ostream>());
    DiagnosticOptions DiagOpts;
    Compiler.setVirtualFileSystem(llvm::vfs::getRealFileSystem());
    Compiler.createDiagnostics(
        new TextDiagnosticPrinter(llvm::nulls(), DiagOpts), true);
    Compiler.setVerboseOutputStream(std::move(VerboseStream));

    Success = ExecuteCompilerInvocation(&Compiler);
  }
  EXPECT_FALSE(Success);
  EXPECT_TRUE(!VerboseBuffer.empty());
  EXPECT_TRUE(StringRef(VerboseBuffer.data()).contains("errors generated"));
}

TEST(FrontendOutputTests, TestVerboseOutputStreamOwnedNotLeaked) {
  CompilerInstance Compiler;
  Compiler.setVerboseOutputStream(std::make_unique<raw_null_ostream>());

  // Trust leak sanitizer bots to catch a leak here.
  Compiler.setVerboseOutputStream(llvm::nulls());
}

} // anonymous namespace
