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
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <string>

namespace clang {
namespace tooling {

namespace {

/// Prints out all of the gathered dependencies into a string.
class TestFileCollector : public DependencyFileGenerator {
public:
  TestFileCollector(DependencyOutputOptions &Opts,
                    std::vector<std::string> &Deps)
      : DependencyFileGenerator(Opts), Deps(Deps) {}

  void finishedMainFile(DiagnosticsEngine &Diags) override {
    Deps = getDependencies();
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
  std::string HeaderPath = llvm::formatv("{0}root{0}header.h", Sept);
  std::string SymlinkPath = llvm::formatv("{0}root{0}symlink.h", Sept);
  std::string TestPath = llvm::formatv("{0}root{0}test.cpp", Sept);

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

} // end namespace tooling
} // end namespace clang
