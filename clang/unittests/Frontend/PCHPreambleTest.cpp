//====-- unittests/Frontend/PCHPreambleTest.cpp - FrontendAction tests ---====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;

namespace {

class ReadCountingInMemoryFileSystem : public vfs::InMemoryFileSystem
{
  std::map<std::string, unsigned> ReadCounts;

public:
  ErrorOr<std::unique_ptr<vfs::File>> openFileForRead(const Twine &Path) override
  {
    SmallVector<char, 128> PathVec;
    Path.toVector(PathVec);
    llvm::sys::path::remove_dots(PathVec, true);
    ++ReadCounts[std::string(PathVec.begin(), PathVec.end())];
    return InMemoryFileSystem::openFileForRead(Path);
  }

  unsigned GetReadCount(const Twine &Path) const
  {
    auto it = ReadCounts.find(Path.str());
    return it == ReadCounts.end() ? 0 : it->second;
  }
};

class PCHPreambleTest : public ::testing::Test {
  IntrusiveRefCntPtr<ReadCountingInMemoryFileSystem> VFS;
  StringMap<std::string> RemappedFiles;
  std::shared_ptr<PCHContainerOperations> PCHContainerOpts;
  FileSystemOptions FSOpts;

public:
  void SetUp() override {
    VFS = new ReadCountingInMemoryFileSystem();
    // We need the working directory to be set to something absolute,
    // otherwise it ends up being inadvertently set to the current
    // working directory in the real file system due to a series of
    // unfortunate conditions interacting badly.
    // What's more, this path *must* be absolute on all (real)
    // filesystems, so just '/' won't work (e.g. on Win32).
    VFS->setCurrentWorkingDirectory("//./");
  }

  void TearDown() override {
  }

  void AddFile(const std::string &Filename, const std::string &Contents) {
    ::time_t now;
    ::time(&now);
    VFS->addFile(Filename, now, MemoryBuffer::getMemBufferCopy(Contents, Filename));
  }

  void RemapFile(const std::string &Filename, const std::string &Contents) {
    RemappedFiles[Filename] = Contents;
  }

  std::unique_ptr<ASTUnit> ParseAST(const std::string &EntryFile) {
    PCHContainerOpts = std::make_shared<PCHContainerOperations>();
    std::shared_ptr<CompilerInvocation> CI(new CompilerInvocation);
    CI->getFrontendOpts().Inputs.push_back(
      FrontendInputFile(EntryFile, FrontendOptions::getInputKindForExtension(
        llvm::sys::path::extension(EntryFile).substr(1))));

    CI->getTargetOpts().Triple = "i386-unknown-linux-gnu";

    CI->getPreprocessorOpts().RemappedFileBuffers = GetRemappedFiles();

    PreprocessorOptions &PPOpts = CI->getPreprocessorOpts();
    PPOpts.RemappedFilesKeepOriginalName = true;

    IntrusiveRefCntPtr<DiagnosticsEngine>
      Diags(CompilerInstance::createDiagnostics(new DiagnosticOptions, new DiagnosticConsumer));

    FileManager *FileMgr = new FileManager(FSOpts, VFS);

    std::unique_ptr<ASTUnit> AST = ASTUnit::LoadFromCompilerInvocation(
      CI, PCHContainerOpts, Diags, FileMgr, false, false,
      /*PrecompilePreambleAfterNParses=*/1);
    return AST;
  }

  bool ReparseAST(const std::unique_ptr<ASTUnit> &AST) {
    bool reparseFailed = AST->Reparse(PCHContainerOpts, GetRemappedFiles(), VFS);
    return !reparseFailed;
  }

  unsigned GetFileReadCount(const std::string &Filename) const {
    return VFS->GetReadCount(Filename);
  }

private:
  std::vector<std::pair<std::string, llvm::MemoryBuffer *>>
  GetRemappedFiles() const {
    std::vector<std::pair<std::string, llvm::MemoryBuffer *>> Remapped;
    for (const auto &RemappedFile : RemappedFiles) {
      std::unique_ptr<MemoryBuffer> buf = MemoryBuffer::getMemBufferCopy(
        RemappedFile.second, RemappedFile.first());
      Remapped.emplace_back(RemappedFile.first(), buf.release());
    }
    return Remapped;
  }
};

TEST_F(PCHPreambleTest, ReparseWithOverriddenFileDoesNotInvalidatePreamble) {
  std::string Header1 = "//./header1.h";
  std::string Header2 = "//./header2.h";
  std::string MainName = "//./main.cpp";
  AddFile(Header1, "");
  AddFile(Header2, "#pragma once");
  AddFile(MainName,
    "#include \"//./foo/../header1.h\"\n"
    "#include \"//./foo/../header2.h\"\n"
    "int main() { return ZERO; }");
  RemapFile(Header1, "static const int ZERO = 0;\n");

  std::unique_ptr<ASTUnit> AST(ParseAST(MainName));
  ASSERT_TRUE(AST.get());
  ASSERT_FALSE(AST->getDiagnostics().hasErrorOccurred());

  unsigned initialCounts[] = {
    GetFileReadCount(MainName),
    GetFileReadCount(Header1),
    GetFileReadCount(Header2)
  };

  ASSERT_TRUE(ReparseAST(AST));

  ASSERT_NE(initialCounts[0], GetFileReadCount(MainName));
  ASSERT_EQ(initialCounts[1], GetFileReadCount(Header1));
  ASSERT_EQ(initialCounts[2], GetFileReadCount(Header2));
}

TEST_F(PCHPreambleTest, ParseWithBom) {
  std::string Header = "//./header.h";
  std::string Main = "//./main.cpp";
  AddFile(Header, "int random() { return 4; }");
  AddFile(Main,
    "\xef\xbb\xbf"
    "#include \"//./header.h\"\n"
    "int main() { return random() -2; }");

  std::unique_ptr<ASTUnit> AST(ParseAST(Main));
  ASSERT_TRUE(AST.get());
  ASSERT_FALSE(AST->getDiagnostics().hasErrorOccurred());

  unsigned HeaderReadCount = GetFileReadCount(Header);

  ASSERT_TRUE(ReparseAST(AST));
  ASSERT_FALSE(AST->getDiagnostics().hasErrorOccurred());
  
  // Check preamble PCH was really reused
  ASSERT_EQ(HeaderReadCount, GetFileReadCount(Header));

  // Remove BOM
  RemapFile(Main,
    "#include \"//./header.h\"\n"
    "int main() { return random() -2; }");

  ASSERT_TRUE(ReparseAST(AST));
  ASSERT_FALSE(AST->getDiagnostics().hasErrorOccurred());

  ASSERT_LE(HeaderReadCount, GetFileReadCount(Header));
  HeaderReadCount = GetFileReadCount(Header);

  // Add BOM back
  RemapFile(Main,
    "\xef\xbb\xbf"
    "#include \"//./header.h\"\n"
    "int main() { return random() -2; }");

  ASSERT_TRUE(ReparseAST(AST));
  ASSERT_FALSE(AST->getDiagnostics().hasErrorOccurred());

  ASSERT_LE(HeaderReadCount, GetFileReadCount(Header));
}

} // anonymous namespace
