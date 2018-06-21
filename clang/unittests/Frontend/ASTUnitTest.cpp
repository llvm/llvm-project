//===- unittests/Frontend/ASTUnitTest.cpp - ASTUnit tests -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

TEST(ASTUnit, SaveLoadPreservesLangOptionsInPrintingPolicy) {
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

  int FD;
  llvm::SmallString<256> InputFileName;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile("ast-unit", "cpp", FD, InputFileName));
  ToolOutputFile input_file(InputFileName, FD);
  input_file.os() << "";

  const char* Args[] = {"clang", "-xc++", InputFileName.c_str()};

  IntrusiveRefCntPtr<DiagnosticsEngine> Diags =
      CompilerInstance::createDiagnostics(new DiagnosticOptions());

  std::shared_ptr<CompilerInvocation> CInvok =
      createInvocationFromCommandLine(Args, Diags);

  if (!CInvok)
    FAIL() << "could not create compiler invocation";

  FileManager *FileMgr =
      new FileManager(FileSystemOptions(), vfs::getRealFileSystem());
  auto PCHContainerOps = std::make_shared<PCHContainerOperations>();

  std::unique_ptr<ASTUnit> AST = ASTUnit::LoadFromCompilerInvocation(
      CInvok, PCHContainerOps, Diags, FileMgr);

  if (!AST)
    FAIL() << "failed to create ASTUnit";

  EXPECT_FALSE(AST->getASTContext().getPrintingPolicy().UseVoidForZeroParams);

  llvm::SmallString<256> ASTFileName;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile("ast-unit", "ast", FD, ASTFileName));
  ToolOutputFile ast_file(ASTFileName, FD);
  AST->Save(ASTFileName.str());

  EXPECT_TRUE(llvm::sys::fs::exists(ASTFileName));

  std::unique_ptr<ASTUnit> AU = ASTUnit::LoadFromASTFile(
      ASTFileName.str(), PCHContainerOps->getRawReader(), ASTUnit::LoadEverything, Diags,
      FileSystemOptions(), /*UseDebugInfo=*/false);

  if (!AU)
    FAIL() << "failed to load ASTUnit";

  EXPECT_FALSE(AU->getASTContext().getPrintingPolicy().UseVoidForZeroParams);
}

} // anonymous namespace
