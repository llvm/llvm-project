//===- unittests/Frontend/ASTUnitTest.cpp - ASTUnit tests -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <fstream>

#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/PCHContainerOperations.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;

namespace {

class ASTUnitTest : public ::testing::Test {
protected:
  int FD;
  llvm::SmallString<256> InputFileName;
  std::unique_ptr<ToolOutputFile> input_file;
  IntrusiveRefCntPtr<DiagnosticsEngine> Diags;
  std::shared_ptr<CompilerInvocation> CInvok;
  std::shared_ptr<PCHContainerOperations> PCHContainerOps;

  std::unique_ptr<ASTUnit> createASTUnit(bool isVolatile) {
    EXPECT_FALSE(llvm::sys::fs::createTemporaryFile("ast-unit", "cpp", FD,
                                                    InputFileName));
    input_file = llvm::make_unique<ToolOutputFile>(InputFileName, FD);
    input_file->os() << "";

    const char *Args[] = {"clang", "-xc++", InputFileName.c_str()};

    Diags = CompilerInstance::createDiagnostics(new DiagnosticOptions());

    CInvok = createInvocationFromCommandLine(Args, Diags);

    if (!CInvok)
      return nullptr;

    FileManager *FileMgr =
        new FileManager(FileSystemOptions(), vfs::getRealFileSystem());
    PCHContainerOps = std::make_shared<PCHContainerOperations>();

    return ASTUnit::LoadFromCompilerInvocation(
        CInvok, PCHContainerOps, Diags, FileMgr, false, CaptureDiagsKind::None,
        0, TU_Complete, false, false, isVolatile);
  }
};

TEST_F(ASTUnitTest, SaveLoadPreservesLangOptionsInPrintingPolicy) {
  // Check that the printing policy is restored with the correct language
  // options when loading an ASTUnit from a file.  To this end, an ASTUnit
  // for a C++ translation unit is set up and written to a temporary file.

  // By default `UseVoidForZeroParams` is true for non-C++ language options,
  // thus we can check this field after loading the ASTUnit to deduce whether
  // the correct (C++) language options were used when setting up the printing
  // policy.

  {
    PrintingPolicy PolicyWithDefaultLangOpt(LangOptions{});
    EXPECT_TRUE(PolicyWithDefaultLangOpt.UseVoidForZeroParams);
  }

  std::unique_ptr<ASTUnit> AST = createASTUnit(false);

  if (!AST)
    FAIL() << "failed to create ASTUnit";

  EXPECT_FALSE(AST->getASTContext().getPrintingPolicy().UseVoidForZeroParams);

  llvm::SmallString<256> ASTFileName;
  ASSERT_FALSE(
      llvm::sys::fs::createTemporaryFile("ast-unit", "ast", FD, ASTFileName));
  ToolOutputFile ast_file(ASTFileName, FD);
  AST->Save(ASTFileName.str());

  EXPECT_TRUE(llvm::sys::fs::exists(ASTFileName));

  std::unique_ptr<ASTUnit> AU = ASTUnit::LoadFromASTFile(
      ASTFileName.str(), PCHContainerOps->getRawReader(),
      ASTUnit::LoadEverything, Diags, FileSystemOptions(),
      /*UseDebugInfo=*/false);

  if (!AU)
    FAIL() << "failed to load ASTUnit";

  EXPECT_FALSE(AU->getASTContext().getPrintingPolicy().UseVoidForZeroParams);
}

TEST_F(ASTUnitTest, GetBufferForFileMemoryMapping) {
  std::unique_ptr<ASTUnit> AST = createASTUnit(true);

  if (!AST)
    FAIL() << "failed to create ASTUnit";

  std::unique_ptr<llvm::MemoryBuffer> memoryBuffer =
      AST->getBufferForFile(InputFileName);

  EXPECT_NE(memoryBuffer->getBufferKind(),
            llvm::MemoryBuffer::MemoryBuffer_MMap);
}

} // anonymous namespace
