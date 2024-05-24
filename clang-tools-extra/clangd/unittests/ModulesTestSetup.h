//===-- ModulesTestSetup.h - Setup the module test environment --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Compiler.h"
#include "support/ThreadsafeFS.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
class ModuleTestSetup : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory("modules-test", TestDir));
  }

  void TearDown() override { llvm::sys::fs::remove_directories(TestDir); }

public:
  // Add files to the working testing directory and repalce all the
  // `__DIR__` to TestDir.
  void addFile(llvm::StringRef Path, llvm::StringRef Contents) {
    ASSERT_FALSE(llvm::sys::path::is_absolute(Path));

    SmallString<256> AbsPath(TestDir);
    llvm::sys::path::append(AbsPath, Path);

    ASSERT_FALSE(llvm::sys::fs::create_directories(
        llvm::sys::path::parent_path(AbsPath)));

    std::error_code EC;
    llvm::raw_fd_ostream OS(AbsPath, EC);
    ASSERT_FALSE(EC);

    std::size_t Pos = Contents.find("__DIR__");
    while (Pos != llvm::StringRef::npos) {
      OS << Contents.take_front(Pos);
      OS << TestDir;
      Contents = Contents.drop_front(Pos + sizeof("__DIR__") - 1);
      Pos = Contents.find("__DIR__");
    }

    OS << Contents;
  }

  // Get the absolute path for file specified by Path under testing working
  // directory.
  std::string getFullPath(llvm::StringRef Path) {
    SmallString<128> Result(TestDir);
    llvm::sys::path::append(Result, Path);
    EXPECT_TRUE(llvm::sys::fs::exists(Result.str()));
    return Result.str().str();
  }

  std::unique_ptr<GlobalCompilationDatabase> getGlobalCompilationDatabase() {
    // The compilation flags with modules are much complex so it looks better
    // to use DirectoryBasedGlobalCompilationDatabase than a mocked compilation
    // database.
    DirectoryBasedGlobalCompilationDatabase::Options Opts(TFS);
    return std::make_unique<DirectoryBasedGlobalCompilationDatabase>(Opts);
  }

  ParseInputs getInputs(llvm::StringRef FileName,
                        const GlobalCompilationDatabase &CDB) {
    std::string FullPathName = getFullPath(FileName);

    ParseInputs Inputs;
    std::optional<tooling::CompileCommand> Cmd =
        CDB.getCompileCommand(FullPathName);
    EXPECT_TRUE(Cmd);
    Inputs.CompileCommand = std::move(*Cmd);
    Inputs.TFS = &TFS;

    if (auto Contents = TFS.view(TestDir)->getBufferForFile(FullPathName))
      Inputs.Contents = Contents->get()->getBuffer().str();

    return Inputs;
  }

  std::unique_ptr<CompilerInvocation>
  getCompilerInvocation(const ParseInputs &Inputs) {
    std::vector<std::string> CC1Args;
    return buildCompilerInvocation(Inputs, DiagConsumer, &CC1Args);
  }

  SmallString<256> TestDir;
  // Noticed MockFS but its member variable 'OverlayRealFileSystemForModules'
  // implies that it will better to use RealThreadsafeFS directly.
  RealThreadsafeFS TFS;

  DiagnosticConsumer DiagConsumer;
};
} // namespace clangd
} // namespace clang
