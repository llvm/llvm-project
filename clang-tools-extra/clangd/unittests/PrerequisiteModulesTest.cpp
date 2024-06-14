//===--------------- PrerequisiteModulesTests.cpp -------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// FIXME: Skip testing on windows temporarily due to the different escaping
/// code mode.
#ifndef _WIN32

#include "ModulesBuilder.h"
#include "ScanningProjectModules.h"

#include "Annotations.h"
#include "CodeComplete.h"
#include "Compiler.h"
#include "TestTU.h"
#include "support/ThreadsafeFS.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang::clangd {
namespace {

class MockDirectoryCompilationDatabase : public MockCompilationDatabase {
public:
  MockDirectoryCompilationDatabase(StringRef TestDir, const ThreadsafeFS &TFS)
      : MockCompilationDatabase(TestDir),
        MockedCDBPtr(std::make_shared<MockClangCompilationDatabase>(*this)),
        TFS(TFS) {
    this->ExtraClangFlags.push_back("-std=c++20");
    this->ExtraClangFlags.push_back("-c");
  }

  void addFile(llvm::StringRef Path, llvm::StringRef Contents);

  std::unique_ptr<ProjectModules> getProjectModules(PathRef) const override {
    return scanningProjectModules(MockedCDBPtr, TFS);
  }

private:
  class MockClangCompilationDatabase : public tooling::CompilationDatabase {
  public:
    MockClangCompilationDatabase(MockDirectoryCompilationDatabase &MCDB)
        : MCDB(MCDB) {}

    std::vector<tooling::CompileCommand>
    getCompileCommands(StringRef FilePath) const override {
      std::optional<tooling::CompileCommand> Cmd =
          MCDB.getCompileCommand(FilePath);
      EXPECT_TRUE(Cmd);
      return {*Cmd};
    }

    std::vector<std::string> getAllFiles() const override { return Files; }

    void AddFile(StringRef File) { Files.push_back(File.str()); }

  private:
    MockDirectoryCompilationDatabase &MCDB;
    std::vector<std::string> Files;
  };

  std::shared_ptr<MockClangCompilationDatabase> MockedCDBPtr;
  const ThreadsafeFS &TFS;
};

// Add files to the working testing directory and the compilation database.
void MockDirectoryCompilationDatabase::addFile(llvm::StringRef Path,
                                               llvm::StringRef Contents) {
  ASSERT_FALSE(llvm::sys::path::is_absolute(Path));

  SmallString<256> AbsPath(Directory);
  llvm::sys::path::append(AbsPath, Path);

  ASSERT_FALSE(
      llvm::sys::fs::create_directories(llvm::sys::path::parent_path(AbsPath)));

  std::error_code EC;
  llvm::raw_fd_ostream OS(AbsPath, EC);
  ASSERT_FALSE(EC);
  OS << Contents;

  MockedCDBPtr->AddFile(Path);
}

class PrerequisiteModulesTests : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory("modules-test", TestDir));
  }

  void TearDown() override {
    ASSERT_FALSE(llvm::sys::fs::remove_directories(TestDir));
  }

public:
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

  SmallString<256> TestDir;
  // Noticed MockFS but its member variable 'OverlayRealFileSystemForModules'
  // implies that it will better to use RealThreadsafeFS directly.
  RealThreadsafeFS TFS;

  DiagnosticConsumer DiagConsumer;
};

TEST_F(PrerequisiteModulesTests, PrerequisiteModulesTest) {
  MockDirectoryCompilationDatabase CDB(TestDir, TFS);

  CDB.addFile("foo.h", R"cpp(
inline void foo() {}
  )cpp");

  CDB.addFile("M.cppm", R"cpp(
module;
#include "foo.h"
export module M;
  )cpp");

  CDB.addFile("N.cppm", R"cpp(
export module N;
import :Part;
import M;
  )cpp");

  CDB.addFile("N-part.cppm", R"cpp(
// Different name with filename intentionally.
export module N:Part;
  )cpp");

  CDB.addFile("bar.h", R"cpp(
inline void bar() {}
  )cpp");

  CDB.addFile("L.cppm", R"cpp(
module;
#include "bar.h"
export module L;
  )cpp");

  CDB.addFile("NonModular.cpp", R"cpp(
#include "bar.h"
#include "foo.h"
void use() {
  foo();
  bar();
}
  )cpp");

  ModulesBuilder Builder(CDB);

  // NonModular.cpp is not related to modules. So nothing should be built.
  {
    auto NonModularInfo =
        Builder.buildPrerequisiteModulesFor(getFullPath("NonModular.cpp"), TFS);
    EXPECT_TRUE(NonModularInfo);
    auto Invocation =
        buildCompilerInvocation(getInputs("NonModular.cpp", CDB), DiagConsumer);
    EXPECT_TRUE(NonModularInfo->canReuse(*Invocation, TFS.view(TestDir)));
  }

  {
    auto MInfo =
        Builder.buildPrerequisiteModulesFor(getFullPath("M.cppm"), TFS);
    // buildPrerequisiteModulesFor won't built the module itself.
    EXPECT_TRUE(MInfo);
    auto Invocation =
        buildCompilerInvocation(getInputs("M.cppm", CDB), DiagConsumer);
    EXPECT_TRUE(MInfo->canReuse(*Invocation, TFS.view(TestDir)));
  }

  // Module N shouldn't be able to be built.
  auto NInfo = Builder.buildPrerequisiteModulesFor(getFullPath("N.cppm"), TFS);
  EXPECT_TRUE(NInfo);

  ParseInputs NInput = getInputs("N.cppm", CDB);
  std::unique_ptr<CompilerInvocation> Invocation =
      buildCompilerInvocation(NInput, DiagConsumer);
  // Test that `PrerequisiteModules::canReuse` works basically.
  EXPECT_TRUE(NInfo->canReuse(*Invocation, TFS.view(TestDir)));

  // Test that we can still reuse the NInfo after we touch a unrelated file.
  {
    CDB.addFile("L.cppm", R"cpp(
module;
#include "bar.h"
export module L;
export int ll = 43;
  )cpp");
    EXPECT_TRUE(NInfo->canReuse(*Invocation, TFS.view(TestDir)));

    CDB.addFile("bar.h", R"cpp(
inline void bar() {}
inline void bar(int) {}
  )cpp");
    EXPECT_TRUE(NInfo->canReuse(*Invocation, TFS.view(TestDir)));
  }

  // Test that we can't reuse the NInfo after we touch a related file.
  {
    CDB.addFile("M.cppm", R"cpp(
module;
#include "foo.h"
export module M;
export int mm = 44;
  )cpp");
    EXPECT_FALSE(NInfo && NInfo->canReuse(*Invocation, TFS.view(TestDir)));

    NInfo = Builder.buildPrerequisiteModulesFor(getFullPath("N.cppm"), TFS);
    EXPECT_TRUE(NInfo && NInfo->canReuse(*Invocation, TFS.view(TestDir)));

    CDB.addFile("foo.h", R"cpp(
inline void foo() {}
inline void foo(int) {}
  )cpp");
    EXPECT_FALSE(NInfo && NInfo->canReuse(*Invocation, TFS.view(TestDir)));

    NInfo = Builder.buildPrerequisiteModulesFor(getFullPath("N.cppm"), TFS);
    EXPECT_TRUE(NInfo && NInfo->canReuse(*Invocation, TFS.view(TestDir)));
  }

  CDB.addFile("N-part.cppm", R"cpp(
export module N:Part;
// Intentioned to make it uncompilable.
export int NPart = 4LIdjwldijaw
  )cpp");
  EXPECT_FALSE(NInfo && NInfo->canReuse(*Invocation, TFS.view(TestDir)));
  NInfo = Builder.buildPrerequisiteModulesFor(getFullPath("N.cppm"), TFS);
  EXPECT_TRUE(NInfo);
  // So NInfo should be unreusable even after rebuild.
  EXPECT_FALSE(NInfo->canReuse(*Invocation, TFS.view(TestDir)));

  CDB.addFile("N-part.cppm", R"cpp(
export module N:Part;
export int NPart = 43;
  )cpp");
  EXPECT_TRUE(NInfo);
  EXPECT_FALSE(NInfo->canReuse(*Invocation, TFS.view(TestDir)));
  NInfo = Builder.buildPrerequisiteModulesFor(getFullPath("N.cppm"), TFS);
  // So NInfo should be unreusable even after rebuild.
  EXPECT_TRUE(NInfo && NInfo->canReuse(*Invocation, TFS.view(TestDir)));

  // Test that if we changed the modification time of the file, the module files
  // info is still reusable if its content doesn't change.
  CDB.addFile("N-part.cppm", R"cpp(
export module N:Part;
export int NPart = 43;
  )cpp");
  EXPECT_TRUE(NInfo && NInfo->canReuse(*Invocation, TFS.view(TestDir)));

  CDB.addFile("N.cppm", R"cpp(
export module N;
import :Part;
import M;

export int nn = 43;
  )cpp");
  // NInfo should be reusable after we change its content.
  EXPECT_TRUE(NInfo && NInfo->canReuse(*Invocation, TFS.view(TestDir)));

  {
    // Check that
    // `PrerequisiteModules::adjustHeaderSearchOptions(HeaderSearchOptions&)`
    // can replace HeaderSearchOptions correctly.
    ParseInputs NInput = getInputs("N.cppm", CDB);
    std::unique_ptr<CompilerInvocation> NInvocation =
        buildCompilerInvocation(NInput, DiagConsumer);
    HeaderSearchOptions &HSOpts = NInvocation->getHeaderSearchOpts();
    NInfo->adjustHeaderSearchOptions(HSOpts);

    EXPECT_TRUE(HSOpts.PrebuiltModuleFiles.count("M"));
    EXPECT_TRUE(HSOpts.PrebuiltModuleFiles.count("N:Part"));
  }
}

// An End-to-End test for modules.
TEST_F(PrerequisiteModulesTests, ParsedASTTest) {
  MockDirectoryCompilationDatabase CDB(TestDir, TFS);

  CDB.addFile("A.cppm", R"cpp(
export module A;
export void printA();
  )cpp");

  CDB.addFile("Use.cpp", R"cpp(
import A;
)cpp");

  ModulesBuilder Builder(CDB);

  ParseInputs Use = getInputs("Use.cpp", CDB);
  Use.ModulesManager = &Builder;

  std::unique_ptr<CompilerInvocation> CI =
      buildCompilerInvocation(Use, DiagConsumer);
  EXPECT_TRUE(CI);

  auto Preamble =
      buildPreamble(getFullPath("Use.cpp"), *CI, Use, /*InMemory=*/true,
                    /*Callback=*/nullptr);
  EXPECT_TRUE(Preamble);
  EXPECT_TRUE(Preamble->RequiredModules);

  auto AST = ParsedAST::build(getFullPath("Use.cpp"), Use, std::move(CI), {},
                              Preamble);
  EXPECT_TRUE(AST);

  const NamedDecl &D = findDecl(*AST, "printA");
  EXPECT_TRUE(D.isFromASTFile());
}

} // namespace
} // namespace clang::clangd

#endif
