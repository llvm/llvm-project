//====-- unittests/Frontend/ReparseWorkingDirTest.cpp - FrontendAction tests =//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;

namespace {
class ReparseWorkingDirTest : public ::testing::Test {
  IntrusiveRefCntPtr<vfs::InMemoryFileSystem> VFS;
  std::shared_ptr<PCHContainerOperations> PCHContainerOpts;

public:
  void SetUp() override { VFS = new vfs::InMemoryFileSystem(); }
  void TearDown() override {}

  void setWorkingDirectory(StringRef Path) {
    VFS->setCurrentWorkingDirectory(Path);
  }

  void AddFile(const std::string &Filename, const std::string &Contents) {
    ::time_t now;
    ::time(&now);
    VFS->addFile(Filename, now,
                 MemoryBuffer::getMemBufferCopy(Contents, Filename));
  }

  std::unique_ptr<ASTUnit> ParseAST(StringRef EntryFile) {
    PCHContainerOpts = std::make_shared<PCHContainerOperations>();
    auto CI = std::make_shared<CompilerInvocation>();
    CI->getFrontendOpts().Inputs.push_back(FrontendInputFile(
        EntryFile, FrontendOptions::getInputKindForExtension(
                       llvm::sys::path::extension(EntryFile).substr(1))));

    CI->getHeaderSearchOpts().AddPath("headers",
                                      frontend::IncludeDirGroup::Quoted,
                                      /*isFramework*/ false,
                                      /*IgnoreSysRoot*/ false);

    CI->getFileSystemOpts().WorkingDir = *VFS->getCurrentWorkingDirectory();
    CI->getTargetOpts().Triple = "i386-unknown-linux-gnu";

    IntrusiveRefCntPtr<DiagnosticsEngine> Diags(
        CompilerInstance::createDiagnostics(*VFS, new DiagnosticOptions,
                                            new DiagnosticConsumer));

    FileManager *FileMgr = new FileManager(CI->getFileSystemOpts(), VFS);

    std::unique_ptr<ASTUnit> AST = ASTUnit::LoadFromCompilerInvocation(
        CI, PCHContainerOpts, Diags, FileMgr, false, CaptureDiagsKind::None,
        /*PrecompilePreambleAfterNParses=*/1);
    return AST;
  }

  bool ReparseAST(const std::unique_ptr<ASTUnit> &AST) {
    bool reparseFailed =
        AST->Reparse(PCHContainerOpts, /*RemappedFiles*/ {}, VFS);
    return !reparseFailed;
  }
};

TEST_F(ReparseWorkingDirTest, ReparseWorkingDir) {
  // Setup the working directory path.
  SmallString<16> WorkingDir;
#ifdef _WIN32
  WorkingDir = "C:\\";
#else
  WorkingDir = "/";
#endif
  llvm::sys::path::append(WorkingDir, "root");
  setWorkingDirectory(WorkingDir);

  SmallString<32> Header;
  llvm::sys::path::append(Header, WorkingDir, "headers", "header.h");

  SmallString<32> MainName;
  llvm::sys::path::append(MainName, WorkingDir, "main.cpp");

  AddFile(MainName.str().str(), R"cpp(
#include "header.h"
int main() { return foo(); }
)cpp");
  AddFile(Header.str().str(), R"h(
static int foo() { return 0; }
)h");

  // Parse the main file, ensuring we can include the header.
  std::unique_ptr<ASTUnit> AST(ParseAST(MainName.str()));
  ASSERT_TRUE(AST.get());
  ASSERT_FALSE(AST->getDiagnostics().hasErrorOccurred());

  // Reparse and check that the working directory was preserved.
  ASSERT_TRUE(ReparseAST(AST));

  const auto &FM = AST->getFileManager();
  const auto &FS = FM.getVirtualFileSystem();
  ASSERT_EQ(FM.getFileSystemOpts().WorkingDir, WorkingDir);
  ASSERT_EQ(*FS.getCurrentWorkingDirectory(), WorkingDir);
}

} // end anonymous namespace
