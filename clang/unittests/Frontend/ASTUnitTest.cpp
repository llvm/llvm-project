//===- unittests/Frontend/ASTUnitTest.cpp - ASTUnit tests -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <fstream>

#include "clang/Basic/FileManager.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/PCHContainerOperations.h"
#include "clang/Lex/HeaderSearch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/VirtualFileSystem.h"
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
    input_file = std::make_unique<ToolOutputFile>(InputFileName, FD);
    input_file->os() << "";

    const char *Args[] = {"clang", "-xc++", InputFileName.c_str()};

    auto VFS = llvm::vfs::getRealFileSystem();
    Diags = CompilerInstance::createDiagnostics(*VFS, new DiagnosticOptions());

    CreateInvocationOptions CIOpts;
    CIOpts.Diags = Diags;
    CIOpts.VFS = VFS;
    CInvok = createInvocation(Args, std::move(CIOpts));

    if (!CInvok)
      return nullptr;

    FileManager *FileMgr = new FileManager(FileSystemOptions(), VFS);
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
  HeaderSearchOptions HSOpts;

  std::unique_ptr<ASTUnit> AU = ASTUnit::LoadFromASTFile(
      ASTFileName, PCHContainerOps->getRawReader(), ASTUnit::LoadEverything,
      Diags, FileSystemOptions(), HSOpts);

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

TEST_F(ASTUnitTest, ModuleTextualHeader) {
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFs =
      new llvm::vfs::InMemoryFileSystem();
  InMemoryFs->addFile("test.cpp", 0, llvm::MemoryBuffer::getMemBuffer(R"cpp(
      #include "Textual.h"
      void foo() {}
    )cpp"));
  InMemoryFs->addFile("m.modulemap", 0, llvm::MemoryBuffer::getMemBuffer(R"cpp(
      module M {
        module Textual {
          textual header "Textual.h"
        }
      }
    )cpp"));
  InMemoryFs->addFile("Textual.h", 0, llvm::MemoryBuffer::getMemBuffer(R"cpp(
      void foo();
    )cpp"));

  const char *Args[] = {"clang", "test.cpp", "-fmodule-map-file=m.modulemap",
                        "-fmodule-name=M"};
  Diags =
      CompilerInstance::createDiagnostics(*InMemoryFs, new DiagnosticOptions());
  CreateInvocationOptions CIOpts;
  CIOpts.Diags = Diags;
  CInvok = createInvocation(Args, std::move(CIOpts));
  ASSERT_TRUE(CInvok);

  FileManager *FileMgr = new FileManager(FileSystemOptions(), InMemoryFs);
  PCHContainerOps = std::make_shared<PCHContainerOperations>();

  auto AU = ASTUnit::LoadFromCompilerInvocation(
      CInvok, PCHContainerOps, Diags, FileMgr, false, CaptureDiagsKind::None, 1,
      TU_Complete, false, false, false);
  ASSERT_TRUE(AU);
  auto File = AU->getFileManager().getFileRef("Textual.h", false, false);
  ASSERT_TRUE(bool(File));
  // Verify that we do not crash here.
  EXPECT_TRUE(
      AU->getPreprocessor().getHeaderSearchInfo().getExistingFileInfo(*File));
}

TEST_F(ASTUnitTest, LoadFromCommandLineEarlyError) {
  EXPECT_FALSE(
      llvm::sys::fs::createTemporaryFile("ast-unit", "c", FD, InputFileName));
  input_file = std::make_unique<ToolOutputFile>(InputFileName, FD);
  input_file->os() << "";

  const char *Args[] = {"clang", "-target", "foobar", InputFileName.c_str()};

  auto Diags = CompilerInstance::createDiagnostics(
      *llvm::vfs::getRealFileSystem(), new DiagnosticOptions());
  auto PCHContainerOps = std::make_shared<PCHContainerOperations>();
  std::unique_ptr<clang::ASTUnit> ErrUnit;

  std::unique_ptr<ASTUnit> AST = ASTUnit::LoadFromCommandLine(
      &Args[0], &Args[4], PCHContainerOps, Diags, "", false, "", false,
      CaptureDiagsKind::All, {}, true, 0, TU_Complete, false, false, false,
      SkipFunctionBodiesScope::None, false, true, false, false, std::nullopt,
      &ErrUnit, nullptr);

  ASSERT_EQ(AST, nullptr);
  ASSERT_NE(ErrUnit, nullptr);
  ASSERT_TRUE(Diags->hasErrorOccurred());
  ASSERT_NE(ErrUnit->stored_diag_size(), 0U);
}

TEST_F(ASTUnitTest, LoadFromCommandLineWorkingDirectory) {
  EXPECT_FALSE(
      llvm::sys::fs::createTemporaryFile("bar", "c", FD, InputFileName));
  auto Input = std::make_unique<ToolOutputFile>(InputFileName, FD);
  Input->os() << "";

  SmallString<128> WorkingDir;
  ASSERT_FALSE(sys::fs::createUniqueDirectory("foo", WorkingDir));
  const char *Args[] = {"clang", "-working-directory", WorkingDir.c_str(),
                        InputFileName.c_str()};

  auto Diags = CompilerInstance::createDiagnostics(
      *llvm::vfs::getRealFileSystem(), new DiagnosticOptions());
  auto PCHContainerOps = std::make_shared<PCHContainerOperations>();
  std::unique_ptr<clang::ASTUnit> ErrUnit;

  std::unique_ptr<ASTUnit> AST = ASTUnit::LoadFromCommandLine(
      &Args[0], &Args[4], PCHContainerOps, Diags, "", false, "", false,
      CaptureDiagsKind::All, {}, true, 0, TU_Complete, false, false, false,
      SkipFunctionBodiesScope::None, false, true, false, false, std::nullopt,
      &ErrUnit, nullptr);

  ASSERT_NE(AST, nullptr);
  ASSERT_FALSE(Diags->hasErrorOccurred());

  // Make sure '-working-directory' sets both the FileSystemOpts and underlying
  // VFS working directory.
  const auto &FM = AST->getFileManager();
  const auto &VFS = FM.getVirtualFileSystem();
  ASSERT_EQ(*VFS.getCurrentWorkingDirectory(), WorkingDir.str());
  ASSERT_EQ(FM.getFileSystemOpts().WorkingDir, WorkingDir.str());
}

} // anonymous namespace
