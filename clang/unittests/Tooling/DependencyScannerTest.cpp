//===- unittest/Tooling/ToolingTest.cpp - Tooling unit tests --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclGroup.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningTool.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <string>

using namespace clang;
using namespace tooling;
using namespace dependencies;

namespace {

/// Prints out all of the gathered dependencies into a string.
class TestFileCollector : public DependencyFileGenerator {
public:
  TestFileCollector(DependencyOutputOptions &Opts,
                    std::vector<std::string> &Deps)
      : DependencyFileGenerator(Opts), Deps(Deps) {}

  void finishedMainFile(DiagnosticsEngine &Diags) override {
    auto NewDeps = getDependencies();
    Deps.insert(Deps.end(), NewDeps.begin(), NewDeps.end());
  }

private:
  std::vector<std::string> &Deps;
};

class TestDependencyScanningAction : public tooling::ToolAction {
public:
  TestDependencyScanningAction(std::vector<std::string> &Deps) : Deps(Deps) {}

  bool runInvocation(std::shared_ptr<CompilerInvocation> Invocation,
                     FileManager *FileMgr,
                     std::shared_ptr<PCHContainerOperations> PCHContainerOps,
                     DiagnosticConsumer *DiagConsumer) override {
    CompilerInstance Compiler(std::move(PCHContainerOps));
    Compiler.setInvocation(std::move(Invocation));
    Compiler.setFileManager(FileMgr);

    Compiler.createDiagnostics(DiagConsumer, /*ShouldOwnClient=*/false);
    if (!Compiler.hasDiagnostics())
      return false;

    Compiler.createSourceManager(*FileMgr);
    Compiler.addDependencyCollector(std::make_shared<TestFileCollector>(
        Compiler.getInvocation().getDependencyOutputOpts(), Deps));

    auto Action = std::make_unique<PreprocessOnlyAction>();
    return Compiler.ExecuteAction(*Action);
  }

private:
  std::vector<std::string> &Deps;
};

} // namespace

TEST(DependencyScanner, ScanDepsReuseFilemanager) {
  std::vector<std::string> Compilation = {"-c", "-E", "-MT", "test.cpp.o"};
  StringRef CWD = "/root";
  FixedCompilationDatabase CDB(CWD, Compilation);

  auto VFS = new llvm::vfs::InMemoryFileSystem();
  VFS->setCurrentWorkingDirectory(CWD);
  auto Sept = llvm::sys::path::get_separator();
  std::string HeaderPath =
      std::string(llvm::formatv("{0}root{0}header.h", Sept));
  std::string SymlinkPath =
      std::string(llvm::formatv("{0}root{0}symlink.h", Sept));
  std::string TestPath = std::string(llvm::formatv("{0}root{0}test.cpp", Sept));

  VFS->addFile(HeaderPath, 0, llvm::MemoryBuffer::getMemBuffer("\n"));
  VFS->addHardLink(SymlinkPath, HeaderPath);
  VFS->addFile(TestPath, 0,
               llvm::MemoryBuffer::getMemBuffer(
                   "#include \"symlink.h\"\n#include \"header.h\"\n"));

  ClangTool Tool(CDB, {"test.cpp"}, std::make_shared<PCHContainerOperations>(),
                 VFS);
  Tool.clearArgumentsAdjusters();
  std::vector<std::string> Deps;
  TestDependencyScanningAction Action(Deps);
  Tool.run(&Action);
  using llvm::sys::path::convert_to_slash;
  // The first invocation should return dependencies in order of access.
  ASSERT_EQ(Deps.size(), 3u);
  EXPECT_EQ(convert_to_slash(Deps[0]), "/root/test.cpp");
  EXPECT_EQ(convert_to_slash(Deps[1]), "/root/symlink.h");
  EXPECT_EQ(convert_to_slash(Deps[2]), "/root/header.h");

  // The file manager should still have two FileEntries, as one file is a
  // hardlink.
  FileManager &Files = Tool.getFiles();
  EXPECT_EQ(Files.getNumUniqueRealFiles(), 2u);

  Deps.clear();
  Tool.run(&Action);
  // The second invocation should have the same order of dependencies.
  ASSERT_EQ(Deps.size(), 3u);
  EXPECT_EQ(convert_to_slash(Deps[0]), "/root/test.cpp");
  EXPECT_EQ(convert_to_slash(Deps[1]), "/root/symlink.h");
  EXPECT_EQ(convert_to_slash(Deps[2]), "/root/header.h");

  EXPECT_EQ(Files.getNumUniqueRealFiles(), 2u);
}

TEST(DependencyScanner, ScanDepsReuseFilemanagerSkippedFile) {
  std::vector<std::string> Compilation = {"-c", "-E", "-MT", "test.cpp.o"};
  StringRef CWD = "/root";
  FixedCompilationDatabase CDB(CWD, Compilation);

  auto VFS = new llvm::vfs::InMemoryFileSystem();
  VFS->setCurrentWorkingDirectory(CWD);
  auto Sept = llvm::sys::path::get_separator();
  std::string HeaderPath =
      std::string(llvm::formatv("{0}root{0}header.h", Sept));
  std::string SymlinkPath =
      std::string(llvm::formatv("{0}root{0}symlink.h", Sept));
  std::string TestPath = std::string(llvm::formatv("{0}root{0}test.cpp", Sept));
  std::string Test2Path =
      std::string(llvm::formatv("{0}root{0}test2.cpp", Sept));

  VFS->addFile(HeaderPath, 0,
               llvm::MemoryBuffer::getMemBuffer("#pragma once\n"));
  VFS->addHardLink(SymlinkPath, HeaderPath);
  VFS->addFile(TestPath, 0,
               llvm::MemoryBuffer::getMemBuffer(
                   "#include \"header.h\"\n#include \"symlink.h\"\n"));
  VFS->addFile(Test2Path, 0,
               llvm::MemoryBuffer::getMemBuffer(
                   "#include \"symlink.h\"\n#include \"header.h\"\n"));

  ClangTool Tool(CDB, {"test.cpp", "test2.cpp"},
                 std::make_shared<PCHContainerOperations>(), VFS);
  Tool.clearArgumentsAdjusters();
  std::vector<std::string> Deps;
  TestDependencyScanningAction Action(Deps);
  Tool.run(&Action);
  using llvm::sys::path::convert_to_slash;
  ASSERT_EQ(Deps.size(), 6u);
  EXPECT_EQ(convert_to_slash(Deps[0]), "/root/test.cpp");
  EXPECT_EQ(convert_to_slash(Deps[1]), "/root/header.h");
  EXPECT_EQ(convert_to_slash(Deps[2]), "/root/symlink.h");
  EXPECT_EQ(convert_to_slash(Deps[3]), "/root/test2.cpp");
  EXPECT_EQ(convert_to_slash(Deps[4]), "/root/symlink.h");
  EXPECT_EQ(convert_to_slash(Deps[5]), "/root/header.h");
}

TEST(DependencyScanner, ScanDepsReuseFilemanagerHasInclude) {
  std::vector<std::string> Compilation = {"-c", "-E", "-MT", "test.cpp.o"};
  StringRef CWD = "/root";
  FixedCompilationDatabase CDB(CWD, Compilation);

  auto VFS = new llvm::vfs::InMemoryFileSystem();
  VFS->setCurrentWorkingDirectory(CWD);
  auto Sept = llvm::sys::path::get_separator();
  std::string HeaderPath =
      std::string(llvm::formatv("{0}root{0}header.h", Sept));
  std::string SymlinkPath =
      std::string(llvm::formatv("{0}root{0}symlink.h", Sept));
  std::string TestPath = std::string(llvm::formatv("{0}root{0}test.cpp", Sept));

  VFS->addFile(HeaderPath, 0, llvm::MemoryBuffer::getMemBuffer("\n"));
  VFS->addHardLink(SymlinkPath, HeaderPath);
  VFS->addFile(
      TestPath, 0,
      llvm::MemoryBuffer::getMemBuffer("#if __has_include(\"header.h\") && "
                                       "__has_include(\"symlink.h\")\n#endif"));

  ClangTool Tool(CDB, {"test.cpp", "test.cpp"},
                 std::make_shared<PCHContainerOperations>(), VFS);
  Tool.clearArgumentsAdjusters();
  std::vector<std::string> Deps;
  TestDependencyScanningAction Action(Deps);
  Tool.run(&Action);
  using llvm::sys::path::convert_to_slash;
  ASSERT_EQ(Deps.size(), 6u);
  EXPECT_EQ(convert_to_slash(Deps[0]), "/root/test.cpp");
  EXPECT_EQ(convert_to_slash(Deps[1]), "/root/header.h");
  EXPECT_EQ(convert_to_slash(Deps[2]), "/root/symlink.h");
  EXPECT_EQ(convert_to_slash(Deps[3]), "/root/test.cpp");
  EXPECT_EQ(convert_to_slash(Deps[4]), "/root/header.h");
  EXPECT_EQ(convert_to_slash(Deps[5]), "/root/symlink.h");
}

TEST(DependencyScanner, ScanDepsWithFS) {
  std::vector<std::string> CommandLine = {"clang",
                                          "-target",
                                          "x86_64-apple-macosx10.7",
                                          "-c",
                                          "test.cpp",
                                          "-o"
                                          "test.cpp.o"};
  StringRef CWD = "/root";

  auto VFS = new llvm::vfs::InMemoryFileSystem();
  VFS->setCurrentWorkingDirectory(CWD);
  auto Sept = llvm::sys::path::get_separator();
  std::string HeaderPath =
      std::string(llvm::formatv("{0}root{0}header.h", Sept));
  std::string TestPath = std::string(llvm::formatv("{0}root{0}test.cpp", Sept));

  VFS->addFile(HeaderPath, 0, llvm::MemoryBuffer::getMemBuffer("\n"));
  VFS->addFile(TestPath, 0,
               llvm::MemoryBuffer::getMemBuffer("#include \"header.h\"\n"));

  DependencyScanningService Service(ScanningMode::DependencyDirectivesScan,
                                    ScanningOutputFormat::Make);
  DependencyScanningTool ScanTool(Service, VFS);

  std::string DepFile;
  ASSERT_THAT_ERROR(
      ScanTool.getDependencyFile(CommandLine, CWD).moveInto(DepFile),
      llvm::Succeeded());
  using llvm::sys::path::convert_to_slash;
  EXPECT_EQ(convert_to_slash(DepFile),
            "test.cpp.o: /root/test.cpp /root/header.h\n");
}

TEST(DependencyScanner, ScanDepsWithModuleLookup) {
  std::vector<std::string> CommandLine = {
      "clang",
      "-target",
      "x86_64-apple-macosx10.7",
      "-c",
      "test.m",
      "-o"
      "test.m.o",
      "-fmodules",
      "-I/root/SomeSources",
  };
  StringRef CWD = "/root";

  auto VFS = new llvm::vfs::InMemoryFileSystem();
  VFS->setCurrentWorkingDirectory(CWD);
  auto Sept = llvm::sys::path::get_separator();
  std::string OtherPath =
      std::string(llvm::formatv("{0}root{0}SomeSources{0}other.h", Sept));
  std::string TestPath = std::string(llvm::formatv("{0}root{0}test.m", Sept));

  VFS->addFile(OtherPath, 0, llvm::MemoryBuffer::getMemBuffer("\n"));
  VFS->addFile(TestPath, 0, llvm::MemoryBuffer::getMemBuffer("@import Foo;\n"));

  struct InterceptorFS : llvm::vfs::ProxyFileSystem {
    std::vector<std::string> StatPaths;
    std::vector<std::string> ReadFiles;

    InterceptorFS(IntrusiveRefCntPtr<FileSystem> UnderlyingFS)
        : ProxyFileSystem(UnderlyingFS) {}

    llvm::ErrorOr<llvm::vfs::Status> status(const Twine &Path) override {
      StatPaths.push_back(Path.str());
      return ProxyFileSystem::status(Path);
    }

    llvm::ErrorOr<std::unique_ptr<llvm::vfs::File>>
    openFileForRead(const Twine &Path) override {
      ReadFiles.push_back(Path.str());
      return ProxyFileSystem::openFileForRead(Path);
    }
  };

  auto InterceptFS = llvm::makeIntrusiveRefCnt<InterceptorFS>(VFS);

  DependencyScanningService Service(ScanningMode::DependencyDirectivesScan,
                                    ScanningOutputFormat::Make);
  DependencyScanningTool ScanTool(Service, InterceptFS);

  // This will fail with "fatal error: module 'Foo' not found" but it doesn't
  // matter, the point of the test is to check that files are not read
  // unnecessarily.
  std::string DepFile;
  ASSERT_THAT_ERROR(
      ScanTool.getDependencyFile(CommandLine, CWD).moveInto(DepFile),
      llvm::Failed());

  EXPECT_TRUE(llvm::find(InterceptFS->StatPaths, OtherPath) ==
              InterceptFS->StatPaths.end());
  EXPECT_EQ(InterceptFS->ReadFiles, std::vector<std::string>{"test.m"});
}
