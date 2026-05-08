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

#include "Annotations.h"
#include "CodeComplete.h"
#include "Compiler.h"
#include "ModulesBuilder.h"
#include "ProjectModules.h"
#include "TestTU.h"
#include "support/Path.h"
#include "support/ThreadsafeFS.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Chrono.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang::clangd {
namespace {

MATCHER_P(named, Name, "") { return arg.Name == Name; }

class GlobalScanningCounterProjectModules : public ProjectModules {
public:
  GlobalScanningCounterProjectModules(
      std::unique_ptr<ProjectModules> Underlying, std::atomic<unsigned> &Count)
      : Underlying(std::move(Underlying)), Count(Count) {}

  std::vector<std::string> getRequiredModules(PathRef File) override {
    return Underlying->getRequiredModules(File);
  }

  std::string getModuleNameForSource(PathRef File) override {
    return Underlying->getModuleNameForSource(File);
  }

  void setCommandMangler(CommandMangler Mangler) override {
    Underlying->setCommandMangler(std::move(Mangler));
  }

  std::string getSourceForModuleName(llvm::StringRef ModuleName,
                                     PathRef RequiredSrcFile) override {
    Count++;
    return Underlying->getSourceForModuleName(ModuleName, RequiredSrcFile);
  }

  ModuleNameState getModuleNameState(llvm::StringRef ModuleName) override {
    return Underlying->getModuleNameState(ModuleName);
  }

private:
  std::unique_ptr<ProjectModules> Underlying;
  std::atomic<unsigned> &Count;
};

class PerFileModulesCompilationDatabase : public GlobalCompilationDatabase {
public:
  PerFileModulesCompilationDatabase(StringRef TestDir, const ThreadsafeFS &TFS)
      : Directory(TestDir), TFS(TFS),
        ToolingCDB(std::make_shared<IndexedCompilationDatabase>(*this)) {}

  void addFile(llvm::StringRef Path, llvm::StringRef Contents,
               std::vector<std::string> ExtraFlags = {}) {
    ASSERT_FALSE(llvm::sys::path::is_absolute(Path));

    SmallString<256> AbsPath(Directory);
    llvm::sys::path::append(AbsPath, Path);

    ASSERT_FALSE(llvm::sys::fs::create_directories(
        llvm::sys::path::parent_path(AbsPath)));

    std::error_code EC;
    llvm::raw_fd_ostream OS(AbsPath, EC);
    ASSERT_FALSE(EC);
    OS << Contents;

    std::vector<std::string> CommandLine = {"clang", "-std=c++20", "-c"};
    CommandLine.insert(CommandLine.end(), ExtraFlags.begin(), ExtraFlags.end());
    CommandLine.push_back(std::string(AbsPath));

    Commands[maybeCaseFoldPath(AbsPath)] = tooling::CompileCommand(
        Directory, std::string(AbsPath), std::move(CommandLine), "");
    Files.push_back(std::string(AbsPath));
  }

  std::optional<tooling::CompileCommand>
  getCompileCommand(PathRef File) const override {
    auto It = Commands.find(maybeCaseFoldPath(File));
    if (It == Commands.end())
      return std::nullopt;
    tooling::CompileCommand Cmd = It->second;
    if (llvm::any_of(Cmd.CommandLine, [](llvm::StringRef Arg) {
          return Arg.starts_with("@");
        })) {
      auto FS = llvm::vfs::getRealFileSystem();
      auto Tokenizer = llvm::Triple(llvm::sys::getProcessTriple()).isOSWindows()
                           ? llvm::cl::TokenizeWindowsCommandLine
                           : llvm::cl::TokenizeGNUCommandLine;
      tooling::addExpandedResponseFiles(Cmd.CommandLine, Cmd.Directory,
                                        Tokenizer, *FS);
    }
    return Cmd;
  }

  std::optional<ProjectInfo> getProjectInfo(PathRef) const override {
    return ProjectInfo{std::string(Directory)};
  }

  std::unique_ptr<ProjectModules> getProjectModules(PathRef) const override {
    return clang::clangd::getProjectModules(ToolingCDB, TFS);
  }

private:
  class IndexedCompilationDatabase : public tooling::CompilationDatabase {
  public:
    IndexedCompilationDatabase(const PerFileModulesCompilationDatabase &CDB)
        : CDB(CDB) {}

    std::vector<tooling::CompileCommand>
    getCompileCommands(StringRef FilePath) const override {
      if (auto Cmd = CDB.getCompileCommand(FilePath))
        return {*Cmd};
      return {};
    }

    std::vector<std::string> getAllFiles() const override { return CDB.Files; }

  private:
    const PerFileModulesCompilationDatabase &CDB;
  };

  std::string Directory;
  const ThreadsafeFS &TFS;
  llvm::StringMap<tooling::CompileCommand> Commands;
  std::vector<std::string> Files;
  std::shared_ptr<IndexedCompilationDatabase> ToolingCDB;
};

class ModuleUnitRootCompilationDatabase
    : public PerFileModulesCompilationDatabase {
public:
  ModuleUnitRootCompilationDatabase(StringRef TestDir, const ThreadsafeFS &TFS)
      : PerFileModulesCompilationDatabase(TestDir, TFS) {}

  std::optional<ProjectInfo> getProjectInfo(PathRef File) const override {
    // Treat each module-unit directory as its own project root so tests can
    // verify that the persistent cache follows the providing module unit.
    llvm::SmallString<256> Root(File);
    llvm::sys::path::remove_filename(Root);
    return ProjectInfo{std::string(Root)};
  }
};

class MockDirectoryCompilationDatabase : public MockCompilationDatabase {
public:
  MockDirectoryCompilationDatabase(StringRef TestDir, const ThreadsafeFS &TFS)
      : MockCompilationDatabase(TestDir),
        MockedCDBPtr(std::make_shared<MockClangCompilationDatabase>(*this)),
        TFS(TFS), GlobalScanningCount(0) {
    this->ExtraClangFlags.push_back("-std=c++20");
    this->ExtraClangFlags.push_back("-c");
  }

  void addFile(llvm::StringRef Path, llvm::StringRef Contents);

  std::unique_ptr<ProjectModules> getProjectModules(PathRef) const override {
    return std::make_unique<GlobalScanningCounterProjectModules>(
        clang::clangd::getProjectModules(MockedCDBPtr, TFS),
        GlobalScanningCount);
  }

  unsigned getGlobalScanningCount() const { return GlobalScanningCount; }

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

  mutable std::atomic<unsigned> GlobalScanningCount;
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

  ParseInputs getInputs(llvm::StringRef FileName,
                        const GlobalCompilationDatabase &CDB) {
    std::string FullPathName = getFullPath(FileName);

    ParseInputs Inputs;
    std::optional<tooling::CompileCommand> Cmd =
        CDB.getCompileCommand(FullPathName);
    EXPECT_TRUE(Cmd);
    Inputs.CompileCommand = std::move(*Cmd);
    Inputs.TFS = &FS;

    if (auto Contents = FS.view(TestDir)->getBufferForFile(FullPathName))
      Inputs.Contents = Contents->get()->getBuffer().str();

    return Inputs;
  }

  SmallString<256> TestDir;
  // FIXME: It will be better to use the MockFS if the scanning process and
  // build module process doesn't depend on reading real IO.
  RealThreadsafeFS FS;

  DiagnosticConsumer DiagConsumer;
};

TEST_F(PrerequisiteModulesTests, NonModularTest) {
  MockDirectoryCompilationDatabase CDB(TestDir, FS);

  CDB.addFile("foo.h", R"cpp(
inline void foo() {}
  )cpp");

  CDB.addFile("NonModular.cpp", R"cpp(
#include "foo.h"
void use() {
  foo();
}
  )cpp");

  ModulesBuilder Builder(CDB);

  // NonModular.cpp is not related to modules. So nothing should be built.
  auto NonModularInfo =
      Builder.buildPrerequisiteModulesFor(getFullPath("NonModular.cpp"), FS);
  EXPECT_TRUE(NonModularInfo);

  HeaderSearchOptions HSOpts;
  NonModularInfo->adjustHeaderSearchOptions(HSOpts);
  EXPECT_TRUE(HSOpts.PrebuiltModuleFiles.empty());

  auto Invocation =
      buildCompilerInvocation(getInputs("NonModular.cpp", CDB), DiagConsumer);
  EXPECT_TRUE(NonModularInfo->canReuse(*Invocation, FS.view(TestDir)));
}

TEST_F(PrerequisiteModulesTests, ModuleWithoutDepTest) {
  MockDirectoryCompilationDatabase CDB(TestDir, FS);

  CDB.addFile("foo.h", R"cpp(
inline void foo() {}
  )cpp");

  CDB.addFile("M.cppm", R"cpp(
module;
#include "foo.h"
export module M;
  )cpp");

  ModulesBuilder Builder(CDB);

  auto MInfo = Builder.buildPrerequisiteModulesFor(getFullPath("M.cppm"), FS);
  EXPECT_TRUE(MInfo);

  // Nothing should be built since M doesn't dependent on anything.
  HeaderSearchOptions HSOpts;
  MInfo->adjustHeaderSearchOptions(HSOpts);
  EXPECT_TRUE(HSOpts.PrebuiltModuleFiles.empty());

  auto Invocation =
      buildCompilerInvocation(getInputs("M.cppm", CDB), DiagConsumer);
  EXPECT_TRUE(MInfo->canReuse(*Invocation, FS.view(TestDir)));
}

TEST_F(PrerequisiteModulesTests, ModuleWithArgumentPatch) {
  MockDirectoryCompilationDatabase CDB(TestDir, FS);

  CDB.ExtraClangFlags.push_back("-invalid-unknown-flag");

  CDB.addFile("Dep.cppm", R"cpp(
export module Dep;
  )cpp");

  CDB.addFile("M.cppm", R"cpp(
export module M;
import Dep;
  )cpp");

  // An invalid flag will break the module compilation and the
  // getRequiredModules would return an empty array
  auto ProjectModules = CDB.getProjectModules(getFullPath("M.cppm"));
  EXPECT_TRUE(
      ProjectModules->getRequiredModules(getFullPath("M.cppm")).empty());

  // Set the mangler to filter out the invalid flag
  ProjectModules->setCommandMangler([](tooling::CompileCommand &Command,
                                       PathRef) {
    auto const It = llvm::find(Command.CommandLine, "-invalid-unknown-flag");
    Command.CommandLine.erase(It);
  });

  // And now it returns a non-empty list of required modules since the
  // compilation succeeded
  EXPECT_FALSE(
      ProjectModules->getRequiredModules(getFullPath("M.cppm")).empty());
}

TEST_F(PrerequisiteModulesTests, ModuleWithDepTest) {
  MockDirectoryCompilationDatabase CDB(TestDir, FS);

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
// Different module name with filename intentionally.
export module N:Part;
  )cpp");

  ModulesBuilder Builder(CDB);

  auto NInfo = Builder.buildPrerequisiteModulesFor(getFullPath("N.cppm"), FS);
  EXPECT_TRUE(NInfo);

  ParseInputs NInput = getInputs("N.cppm", CDB);
  std::unique_ptr<CompilerInvocation> Invocation =
      buildCompilerInvocation(NInput, DiagConsumer);
  // Test that `PrerequisiteModules::canReuse` works basically.
  EXPECT_TRUE(NInfo->canReuse(*Invocation, FS.view(TestDir)));

  {
    // Check that
    // `PrerequisiteModules::adjustHeaderSearchOptions(HeaderSearchOptions&)`
    // can appending HeaderSearchOptions correctly.
    HeaderSearchOptions HSOpts;
    NInfo->adjustHeaderSearchOptions(HSOpts);

    EXPECT_TRUE(HSOpts.PrebuiltModuleFiles.count("M"));
    EXPECT_TRUE(HSOpts.PrebuiltModuleFiles.count("N:Part"));
  }

  {
    // Check that
    // `PrerequisiteModules::adjustHeaderSearchOptions(HeaderSearchOptions&)`
    // can replace HeaderSearchOptions correctly.
    HeaderSearchOptions HSOpts;
    HSOpts.PrebuiltModuleFiles["M"] = "incorrect_path";
    HSOpts.PrebuiltModuleFiles["N:Part"] = "incorrect_path";
    NInfo->adjustHeaderSearchOptions(HSOpts);

    EXPECT_TRUE(StringRef(HSOpts.PrebuiltModuleFiles["M"]).ends_with(".pcm"));
    EXPECT_TRUE(
        StringRef(HSOpts.PrebuiltModuleFiles["N:Part"]).ends_with(".pcm"));
  }
}

TEST_F(PrerequisiteModulesTests, ReusabilityTest) {
  MockDirectoryCompilationDatabase CDB(TestDir, FS);

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
// Different module name with filename intentionally.
export module N:Part;
  )cpp");

  ModulesBuilder Builder(CDB);

  auto NInfo = Builder.buildPrerequisiteModulesFor(getFullPath("N.cppm"), FS);
  EXPECT_TRUE(NInfo);
  EXPECT_TRUE(NInfo);

  ParseInputs NInput = getInputs("N.cppm", CDB);
  std::unique_ptr<CompilerInvocation> Invocation =
      buildCompilerInvocation(NInput, DiagConsumer);
  EXPECT_TRUE(NInfo->canReuse(*Invocation, FS.view(TestDir)));

  // Test that we can still reuse the NInfo after we touch a unrelated file.
  {
    CDB.addFile("L.cppm", R"cpp(
module;
#include "foo.h"
export module L;
export int ll = 43;
  )cpp");
    EXPECT_TRUE(NInfo->canReuse(*Invocation, FS.view(TestDir)));

    CDB.addFile("bar.h", R"cpp(
inline void bar() {}
inline void bar(int) {}
  )cpp");
    EXPECT_TRUE(NInfo->canReuse(*Invocation, FS.view(TestDir)));
  }

  // Test that we can't reuse the NInfo after we touch a related file.
  {
    CDB.addFile("M.cppm", R"cpp(
module;
#include "foo.h"
export module M;
export int mm = 44;
  )cpp");
    EXPECT_FALSE(NInfo->canReuse(*Invocation, FS.view(TestDir)));

    NInfo = Builder.buildPrerequisiteModulesFor(getFullPath("N.cppm"), FS);
    EXPECT_TRUE(NInfo->canReuse(*Invocation, FS.view(TestDir)));

    CDB.addFile("foo.h", R"cpp(
inline void foo() {}
inline void foo(int) {}
  )cpp");
    EXPECT_FALSE(NInfo->canReuse(*Invocation, FS.view(TestDir)));

    NInfo = Builder.buildPrerequisiteModulesFor(getFullPath("N.cppm"), FS);
    EXPECT_TRUE(NInfo->canReuse(*Invocation, FS.view(TestDir)));
  }

  CDB.addFile("N-part.cppm", R"cpp(
export module N:Part;
// Intentioned to make it uncompilable.
export int NPart = 4LIdjwldijaw
  )cpp");
  EXPECT_FALSE(NInfo->canReuse(*Invocation, FS.view(TestDir)));
  NInfo = Builder.buildPrerequisiteModulesFor(getFullPath("N.cppm"), FS);
  EXPECT_TRUE(NInfo);
  EXPECT_FALSE(NInfo->canReuse(*Invocation, FS.view(TestDir)));

  CDB.addFile("N-part.cppm", R"cpp(
export module N:Part;
export int NPart = 43;
  )cpp");
  EXPECT_TRUE(NInfo);
  EXPECT_FALSE(NInfo->canReuse(*Invocation, FS.view(TestDir)));
  NInfo = Builder.buildPrerequisiteModulesFor(getFullPath("N.cppm"), FS);
  EXPECT_TRUE(NInfo);
  EXPECT_TRUE(NInfo->canReuse(*Invocation, FS.view(TestDir)));

  // Test that if we changed the modification time of the file, the module files
  // info is still reusable if its content doesn't change.
  CDB.addFile("N-part.cppm", R"cpp(
export module N:Part;
export int NPart = 43;
  )cpp");
  EXPECT_TRUE(NInfo->canReuse(*Invocation, FS.view(TestDir)));

  CDB.addFile("N.cppm", R"cpp(
export module N;
import :Part;
import M;

export int nn = 43;
  )cpp");
  // NInfo should be reusable after we change its content.
  EXPECT_TRUE(NInfo->canReuse(*Invocation, FS.view(TestDir)));
}

TEST_F(PrerequisiteModulesTests, CanReuseWithTransitiveNamedModuleImports) {
  MockDirectoryCompilationDatabase CDB(TestDir, FS);

  CDB.addFile("N.cppm", R"cpp(
export module N;
export inline constexpr int n = 1;
  )cpp");

  CDB.addFile("M.cppm", R"cpp(
export module M;
import N;
export inline constexpr int m = n + 1;
  )cpp");

  CDB.addFile("A.cppm", R"cpp(
export module A;
import M;
export inline constexpr int a = m + 1;
  )cpp");

  CDB.addFile("Use.cpp", R"cpp(
import A;
int use() { return a; }
  )cpp");

  ModulesBuilder Builder(CDB);

  auto UseInfo = Builder.buildPrerequisiteModulesFor(getFullPath("Use.cpp"), FS);
  ASSERT_TRUE(UseInfo);

  HeaderSearchOptions HSOpts;
  UseInfo->adjustHeaderSearchOptions(HSOpts);
  EXPECT_TRUE(HSOpts.PrebuiltModuleFiles.count("A"));
  EXPECT_TRUE(HSOpts.PrebuiltModuleFiles.count("M"));
  EXPECT_TRUE(HSOpts.PrebuiltModuleFiles.count("N"));

  auto Invocation =
      buildCompilerInvocation(getInputs("Use.cpp", CDB), DiagConsumer);
  ASSERT_TRUE(Invocation);

  EXPECT_TRUE(UseInfo->canReuse(*Invocation, FS.view(TestDir)));
}

// An End-to-End test for modules.
TEST_F(PrerequisiteModulesTests, ParsedASTTest) {
  MockDirectoryCompilationDatabase CDB(TestDir, FS);

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

// An end to end test for code complete in modules
TEST_F(PrerequisiteModulesTests, CodeCompleteTest) {
  MockDirectoryCompilationDatabase CDB(TestDir, FS);

  CDB.addFile("A.cppm", R"cpp(
export module A;
export void printA();
  )cpp");

  llvm::StringLiteral UserContents = R"cpp(
import A;
void func() {
  print^
}
)cpp";

  CDB.addFile("Use.cpp", UserContents);
  Annotations Test(UserContents);

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

  auto Result = codeComplete(getFullPath("Use.cpp"), Test.point(),
                             Preamble.get(), Use, {});
  EXPECT_FALSE(Result.Completions.empty());
  EXPECT_EQ(Result.Completions[0].Name, "printA");
}

TEST_F(PrerequisiteModulesTests, SignatureHelpTest) {
  MockDirectoryCompilationDatabase CDB(TestDir, FS);

  CDB.addFile("A.cppm", R"cpp(
export module A;
export void printA(int a);
  )cpp");

  llvm::StringLiteral UserContents = R"cpp(
import A;
void func() {
  printA(^);
}
)cpp";

  CDB.addFile("Use.cpp", UserContents);
  Annotations Test(UserContents);

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

  auto Result = signatureHelp(getFullPath("Use.cpp"), Test.point(), *Preamble,
                              Use, MarkupKind::PlainText);
  EXPECT_FALSE(Result.signatures.empty());
  EXPECT_EQ(Result.signatures[0].label, "printA(int a) -> void");
  EXPECT_EQ(Result.signatures[0].parameters[0].labelString, "int a");
}

TEST_F(PrerequisiteModulesTests, ReusablePrerequisiteModulesTest) {
  MockDirectoryCompilationDatabase CDB(TestDir, FS);

  CDB.addFile("M.cppm", R"cpp(
export module M;
export int M = 43;
  )cpp");
  CDB.addFile("A.cppm", R"cpp(
export module A;
import M;
export int A = 43 + M;
  )cpp");
  CDB.addFile("B.cppm", R"cpp(
export module B;
import M;
export int B = 44 + M;
  )cpp");

  ModulesBuilder Builder(CDB);

  auto AInfo = Builder.buildPrerequisiteModulesFor(getFullPath("A.cppm"), FS);
  EXPECT_TRUE(AInfo);
  auto BInfo = Builder.buildPrerequisiteModulesFor(getFullPath("B.cppm"), FS);
  EXPECT_TRUE(BInfo);
  HeaderSearchOptions HSOptsA(TestDir);
  HeaderSearchOptions HSOptsB(TestDir);
  AInfo->adjustHeaderSearchOptions(HSOptsA);
  BInfo->adjustHeaderSearchOptions(HSOptsB);

  EXPECT_FALSE(HSOptsA.PrebuiltModuleFiles.empty());
  EXPECT_FALSE(HSOptsB.PrebuiltModuleFiles.empty());

  // Check that we're reusing the module files.
  EXPECT_EQ(HSOptsA.PrebuiltModuleFiles, HSOptsB.PrebuiltModuleFiles);

  // Update M.cppm to check if the modules builder can update correctly.
  CDB.addFile("M.cppm", R"cpp(
export module M;
export constexpr int M = 43;
  )cpp");

  ParseInputs AUse = getInputs("A.cppm", CDB);
  AUse.ModulesManager = &Builder;
  std::unique_ptr<CompilerInvocation> AInvocation =
      buildCompilerInvocation(AUse, DiagConsumer);
  EXPECT_FALSE(AInfo->canReuse(*AInvocation, FS.view(TestDir)));

  ParseInputs BUse = getInputs("B.cppm", CDB);
  AUse.ModulesManager = &Builder;
  std::unique_ptr<CompilerInvocation> BInvocation =
      buildCompilerInvocation(BUse, DiagConsumer);
  EXPECT_FALSE(BInfo->canReuse(*BInvocation, FS.view(TestDir)));

  auto NewAInfo =
      Builder.buildPrerequisiteModulesFor(getFullPath("A.cppm"), FS);
  auto NewBInfo =
      Builder.buildPrerequisiteModulesFor(getFullPath("B.cppm"), FS);
  EXPECT_TRUE(NewAInfo);
  EXPECT_TRUE(NewBInfo);
  HeaderSearchOptions NewHSOptsA(TestDir);
  HeaderSearchOptions NewHSOptsB(TestDir);
  NewAInfo->adjustHeaderSearchOptions(NewHSOptsA);
  NewBInfo->adjustHeaderSearchOptions(NewHSOptsB);

  EXPECT_FALSE(NewHSOptsA.PrebuiltModuleFiles.empty());
  EXPECT_FALSE(NewHSOptsB.PrebuiltModuleFiles.empty());

  EXPECT_EQ(NewHSOptsA.PrebuiltModuleFiles, NewHSOptsB.PrebuiltModuleFiles);
  // Persistent cache keeps the published BMI path stable, so verify the new
  // module graph by reuse semantics instead of expecting a different path.
  EXPECT_TRUE(NewAInfo->canReuse(*AInvocation, FS.view(TestDir)));
  EXPECT_TRUE(NewBInfo->canReuse(*BInvocation, FS.view(TestDir)));
}

TEST_F(PrerequisiteModulesTests, ScanningCacheTest) {
  MockDirectoryCompilationDatabase CDB(TestDir, FS);

  CDB.addFile("M.cppm", R"cpp(
export module M;
  )cpp");
  CDB.addFile("A.cppm", R"cpp(
export module A;
import M;
  )cpp");
  CDB.addFile("B.cppm", R"cpp(
export module B;
import M;
  )cpp");

  ModulesBuilder Builder(CDB);

  Builder.buildPrerequisiteModulesFor(getFullPath("A.cppm"), FS);
  Builder.buildPrerequisiteModulesFor(getFullPath("B.cppm"), FS);
  EXPECT_EQ(CDB.getGlobalScanningCount(), 1u);
}

// Test that canReuse detects changes to headers included in module units.
// This verifies that the ASTReader correctly tracks header file dependencies
// in BMI files and that IsModuleFileUpToDate correctly validates them.
TEST_F(PrerequisiteModulesTests, CanReuseWithHeadersInModuleUnit) {
  MockDirectoryCompilationDatabase CDB(TestDir, FS);

  // Create a header file that will be included in a module unit
  CDB.addFile("header1.h", R"cpp(
inline int getValue() { return 42; }
  )cpp");

  // Module M includes header1.h in the global module fragment
  CDB.addFile("M.cppm", R"cpp(
module;
#include "header1.h"
export module M;
export int m_value = getValue();
  )cpp");

  // Module N imports M (similar structure to ReusabilityTest)
  CDB.addFile("N.cppm", R"cpp(
export module N;
import :Part;
import M;
  )cpp");

  // Add a module partition (similar to ReusabilityTest)
  CDB.addFile("N-part.cppm", R"cpp(
export module N:Part;
  )cpp");

  ModulesBuilder Builder(CDB);

  // Build prerequisite modules for N (which depends on M)
  auto NInfo = Builder.buildPrerequisiteModulesFor(getFullPath("N.cppm"), FS);
  ASSERT_TRUE(NInfo);

  ParseInputs NInput = getInputs("N.cppm", CDB);
  std::unique_ptr<CompilerInvocation> Invocation =
      buildCompilerInvocation(NInput, DiagConsumer);

  // Initially, canReuse should return true
  EXPECT_TRUE(NInfo->canReuse(*Invocation, FS.view(TestDir)));

  // Test 1: Modify header1.h (included by M)
  // canReuse should detect this change since M's BMI records header1.h as input
  CDB.addFile("header1.h", R"cpp(
inline int getValue() { return 43; }
  )cpp");
  EXPECT_FALSE(NInfo->canReuse(*Invocation, FS.view(TestDir)));

  // Rebuild and verify canReuse returns true again
  NInfo = Builder.buildPrerequisiteModulesFor(getFullPath("N.cppm"), FS);
  ASSERT_TRUE(NInfo);
  EXPECT_TRUE(NInfo->canReuse(*Invocation, FS.view(TestDir)));

  // Test 2: Modify the module source file itself
  CDB.addFile("M.cppm", R"cpp(
module;
#include "header1.h"
export module M;
export int m_value = getValue();
export int m_new_value = 10;
  )cpp");
  EXPECT_FALSE(NInfo->canReuse(*Invocation, FS.view(TestDir)));

  // Rebuild after module source change
  NInfo = Builder.buildPrerequisiteModulesFor(getFullPath("N.cppm"), FS);
  ASSERT_TRUE(NInfo);
  EXPECT_TRUE(NInfo->canReuse(*Invocation, FS.view(TestDir)));
}

TEST_F(PrerequisiteModulesTests, PrebuiltModuleFileTest) {
  MockDirectoryCompilationDatabase CDB(TestDir, FS);

  CDB.addFile("M.cppm", R"cpp(
export module M;
  )cpp");

  CDB.addFile("U.cpp", R"cpp(
import M;
  )cpp");

  // Use ModulesBuilder to produce the prebuilt module file.
  ModulesBuilder Builder(CDB);
  auto ModuleInfo =
      Builder.buildPrerequisiteModulesFor(getFullPath("U.cpp"), FS);
  HeaderSearchOptions HS(TestDir);
  ModuleInfo->adjustHeaderSearchOptions(HS);

  CDB.ExtraClangFlags.push_back("-fmodule-file=M=" +
                                HS.PrebuiltModuleFiles["M"]);
  ModulesBuilder Builder2(CDB);
  auto ModuleInfo2 =
      Builder2.buildPrerequisiteModulesFor(getFullPath("U.cpp"), FS);
  HeaderSearchOptions HS2(TestDir);
  ModuleInfo2->adjustHeaderSearchOptions(HS2);

  EXPECT_EQ(HS.PrebuiltModuleFiles, HS2.PrebuiltModuleFiles);
}

// Test that prebuilt module files with relative paths are correctly resolved.
// This tests the fix for the issue where clangd couldn't find BMI files when
// the compilation database contained relative paths in -fmodule-file=
// arguments.
TEST_F(PrerequisiteModulesTests, PrebuiltModuleFileWithRelativePath) {
  MockDirectoryCompilationDatabase CDB(TestDir, FS);

  CDB.addFile("M.cppm", R"cpp(
export module M;
export int m_value = 42;
  )cpp");

  CDB.addFile("U.cpp", R"cpp(
import M;
int use() { return m_value; }
  )cpp");

  // Step 1: Build the module file using ModulesBuilder
  ModulesBuilder Builder(CDB);
  auto ModuleInfo =
      Builder.buildPrerequisiteModulesFor(getFullPath("U.cpp"), FS);
  ASSERT_TRUE(ModuleInfo);

  HeaderSearchOptions HS(TestDir);
  ModuleInfo->adjustHeaderSearchOptions(HS);
  ASSERT_EQ(HS.PrebuiltModuleFiles.count("M"), 1u);

  // Get the absolute path of the built module file
  std::string OriginalBMPath = HS.PrebuiltModuleFiles["M"];
  ASSERT_TRUE(llvm::sys::path::is_absolute(OriginalBMPath));
  ASSERT_TRUE(llvm::sys::fs::exists(OriginalBMPath));

  // Step 2: Create a subdirectory in TestDir and copy the BMI there
  SmallString<256> BMSubDir(TestDir);
  llvm::sys::path::append(BMSubDir, "prebuilt_modules");
  ASSERT_FALSE(llvm::sys::fs::create_directories(BMSubDir));

  SmallString<256> NewBMPath(BMSubDir);
  llvm::sys::path::append(NewBMPath, "M.pcm");

  // Copy the BMI file to the new location
  ASSERT_FALSE(llvm::sys::fs::copy_file(OriginalBMPath, NewBMPath));
  ASSERT_TRUE(llvm::sys::fs::exists(NewBMPath));

  // Step 3: Create a relative path from the new absolute path
  std::string RelativeBMPath =
      llvm::StringRef(NewBMPath).drop_front(TestDir.size() + 1).str();
  ASSERT_FALSE(RelativeBMPath.empty());
  ASSERT_TRUE(llvm::sys::path::is_relative(RelativeBMPath));

  // Step 4: Create a new CDB with relative path in -fmodule-file=
  MockDirectoryCompilationDatabase CDBWithRelativePath(TestDir, FS);

  CDBWithRelativePath.addFile("M.cppm", R"cpp(
export module M;
export int m_value = 42;
  )cpp");

  CDBWithRelativePath.addFile("U.cpp", R"cpp(
import M;
int use() { return m_value; }
  )cpp");

  // Use relative path in -fmodule-file= argument
  CDBWithRelativePath.ExtraClangFlags.push_back("-fmodule-file=M=" +
                                                RelativeBMPath);

  // Step 5: Verify that clangd can find and reuse the prebuilt module file
  ModulesBuilder BuilderWithRelativePath(CDBWithRelativePath);
  auto ModuleInfo2 = BuilderWithRelativePath.buildPrerequisiteModulesFor(
      getFullPath("U.cpp"), FS);
  ASSERT_TRUE(ModuleInfo2);

  HeaderSearchOptions HS2(TestDir);
  ModuleInfo2->adjustHeaderSearchOptions(HS2);

  // The module file should be found and the paths should match
  ASSERT_EQ(HS2.PrebuiltModuleFiles.count("M"), 1u);
  EXPECT_EQ(HS2.PrebuiltModuleFiles["M"], std::string(NewBMPath))
      << "Expected absolute path: " << NewBMPath
      << "\nGot: " << HS2.PrebuiltModuleFiles["M"]
      << "\nRelative path used: " << RelativeBMPath;
}

TEST_F(PrerequisiteModulesTests,
       UniqueModuleNameStateResolvedFromCompileCommands) {
  PerFileModulesCompilationDatabase CDB(TestDir, FS);

  SmallString<256> MPcm(TestDir);
  llvm::sys::path::append(MPcm, "build", "M.pcm");

  CDB.addFile("M.cppm", R"cpp(
export module M;
export int value = 1;
  )cpp",
              {"--precompile", "-o", std::string(MPcm)});
  CDB.addFile("A.cpp", R"cpp(
import M;
int useA() { return value; }
  )cpp",
              {"-fmodule-file=M=" + std::string(MPcm)});
  CDB.addFile("B.cpp", R"cpp(
import M;
int useB() { return value; }
  )cpp",
              {"-fmodule-file=M=" + std::string(MPcm)});

  auto ProjectModules = CDB.getProjectModules(getFullPath("A.cpp"));
  ASSERT_TRUE(ProjectModules);

  EXPECT_EQ(ProjectModules->getModuleNameState("M"),
            ProjectModules::ModuleNameState::Unique);
}

TEST_F(PrerequisiteModulesTests,
       DuplicateModuleNamesResolvedFromCompileCommands) {
  PerFileModulesCompilationDatabase CDB(TestDir, FS);

  SmallString<256> APcm(TestDir);
  llvm::sys::path::append(APcm, "build", "a", "M.pcm");
  SmallString<256> BPcm(TestDir);
  llvm::sys::path::append(BPcm, "build", "b", "M.pcm");

  CDB.addFile("a/M.cppm", R"cpp(
export module M;
export int onlyA = 1;
  )cpp",
              {"--precompile", "-o", std::string(APcm)});
  CDB.addFile("b/M.cppm", R"cpp(
export module M;
export int onlyB = 2;
  )cpp",
              {"--precompile", "-o", std::string(BPcm)});
  CDB.addFile("a/Use.cpp", R"cpp(
import M;
int useA() { return onlyA; }
  )cpp",
              {"-fmodule-file=M=" + std::string(APcm)});
  CDB.addFile("b/Use.cpp", R"cpp(
import M;
int useB() { return onlyB; }
  )cpp",
              {"-fmodule-file=M=" + std::string(BPcm)});

  auto ProjectModules = CDB.getProjectModules(getFullPath("a/Use.cpp"));
  ASSERT_TRUE(ProjectModules);
  EXPECT_EQ(ProjectModules->getModuleNameState("M"),
            ProjectModules::ModuleNameState::Multiple);

  EXPECT_EQ(
      ProjectModules->getSourceForModuleName("M", getFullPath("a/Use.cpp")),
      getFullPath("a/M.cppm"));
  EXPECT_EQ(
      ProjectModules->getSourceForModuleName("M", getFullPath("b/Use.cpp")),
      getFullPath("b/M.cppm"));
}

TEST_F(PrerequisiteModulesTests,
       DuplicateModuleNamesResolvedFromResponseFiles) {
  PerFileModulesCompilationDatabase CDB(TestDir, FS);

  SmallString<256> APcm(TestDir);
  llvm::sys::path::append(APcm, "build", "a", "M.pcm");
  SmallString<256> BPcm(TestDir);
  llvm::sys::path::append(BPcm, "build", "b", "M.pcm");

  SmallString<256> RspDir(TestDir);
  llvm::sys::path::append(RspDir, "build", "rsp");
  ASSERT_FALSE(llvm::sys::fs::create_directories(RspDir));

  SmallString<256> AMRsp(RspDir);
  llvm::sys::path::append(AMRsp, "a-m.rsp");
  {
    std::error_code EC;
    llvm::raw_fd_ostream OS(AMRsp, EC);
    ASSERT_FALSE(EC);
    OS << "-x c++-module -fmodule-output=" << APcm;
    OS.close();
  }

  SmallString<256> BMRsp(RspDir);
  llvm::sys::path::append(BMRsp, "b-m.rsp");
  {
    std::error_code EC;
    llvm::raw_fd_ostream OS(BMRsp, EC);
    ASSERT_FALSE(EC);
    OS << "-x c++-module -fmodule-output=" << BPcm;
    OS.close();
  }

  SmallString<256> AUseRsp(RspDir);
  llvm::sys::path::append(AUseRsp, "a-use.rsp");
  {
    std::error_code EC;
    llvm::raw_fd_ostream OS(AUseRsp, EC);
    ASSERT_FALSE(EC);
    OS << "-fmodule-file=M=" << APcm;
    OS.close();
  }

  SmallString<256> BUseRsp(RspDir);
  llvm::sys::path::append(BUseRsp, "b-use.rsp");
  {
    std::error_code EC;
    llvm::raw_fd_ostream OS(BUseRsp, EC);
    ASSERT_FALSE(EC);
    OS << "-fmodule-file=M=" << BPcm;
    OS.close();
  }

  CDB.addFile("a/M.cppm", R"cpp(
export module M;
export int onlyA = 1;
  )cpp",
              {"@" + std::string(AMRsp)});
  CDB.addFile("b/M.cppm", R"cpp(
export module M;
export int onlyB = 2;
  )cpp",
              {"@" + std::string(BMRsp)});
  CDB.addFile("a/Use.cpp", R"cpp(
import M;
int useA() { return onlyA; }
  )cpp",
              {"@" + std::string(AUseRsp)});
  CDB.addFile("b/Use.cpp", R"cpp(
import M;
int useB() { return onlyB; }
  )cpp",
              {"@" + std::string(BUseRsp)});

  auto ProjectModules = CDB.getProjectModules(getFullPath("a/Use.cpp"));
  ASSERT_TRUE(ProjectModules);
  EXPECT_EQ(ProjectModules->getModuleNameState("M"),
            ProjectModules::ModuleNameState::Multiple);

  EXPECT_EQ(
      ProjectModules->getSourceForModuleName("M", getFullPath("a/Use.cpp")),
      getFullPath("a/M.cppm"));
  EXPECT_EQ(
      ProjectModules->getSourceForModuleName("M", getFullPath("b/Use.cpp")),
      getFullPath("b/M.cppm"));
}

TEST_F(PrerequisiteModulesTests, DuplicateModuleNamesKeepSeparateBMICache) {
  PerFileModulesCompilationDatabase CDB(TestDir, FS);

  SmallString<256> APcm(TestDir);
  llvm::sys::path::append(APcm, "build", "a", "M.pcm");
  SmallString<256> BPcm(TestDir);
  llvm::sys::path::append(BPcm, "build", "b", "M.pcm");

  CDB.addFile("a/M.cppm", R"cpp(
export module M;
export int onlyA = 1;
  )cpp",
              {"--precompile", "-o", std::string(APcm)});
  CDB.addFile("b/M.cppm", R"cpp(
export module M;
export int onlyB = 2;
  )cpp",
              {"--precompile", "-o", std::string(BPcm)});
  CDB.addFile("a/Use.cpp", R"cpp(
import M;
int useA() { return onlyA; }
  )cpp",
              {"-fmodule-file=M=" + std::string(APcm)});
  CDB.addFile("b/Use.cpp", R"cpp(
import M;
int useB() { return onlyB; }
  )cpp",
              {"-fmodule-file=M=" + std::string(BPcm)});

  ModulesBuilder Builder(CDB);

  auto AInfo =
      Builder.buildPrerequisiteModulesFor(getFullPath("a/Use.cpp"), FS);
  auto BInfo =
      Builder.buildPrerequisiteModulesFor(getFullPath("b/Use.cpp"), FS);
  ASSERT_TRUE(AInfo);
  ASSERT_TRUE(BInfo);

  HeaderSearchOptions HSA(TestDir);
  HeaderSearchOptions HSB(TestDir);
  AInfo->adjustHeaderSearchOptions(HSA);
  BInfo->adjustHeaderSearchOptions(HSB);
  ASSERT_EQ(HSA.PrebuiltModuleFiles.count("M"), 1u);
  ASSERT_EQ(HSB.PrebuiltModuleFiles.count("M"), 1u);
  EXPECT_NE(HSA.PrebuiltModuleFiles["M"], HSB.PrebuiltModuleFiles["M"]);

  auto UseA = getInputs("a/Use.cpp", CDB);
  UseA.ModulesManager = &Builder;
  auto CIA = buildCompilerInvocation(UseA, DiagConsumer);
  ASSERT_TRUE(CIA);
  auto PreambleA = buildPreamble(getFullPath("a/Use.cpp"), *CIA, UseA,
                                 /*InMemory=*/true, /*Callback=*/nullptr);
  ASSERT_TRUE(PreambleA);
  auto ASTA = ParsedAST::build(getFullPath("a/Use.cpp"), UseA, std::move(CIA),
                               {}, PreambleA);
  ASSERT_TRUE(ASTA);
  EXPECT_TRUE(findDecl(*ASTA, "onlyA").isFromASTFile());

  auto UseB = getInputs("b/Use.cpp", CDB);
  UseB.ModulesManager = &Builder;
  auto CIB = buildCompilerInvocation(UseB, DiagConsumer);
  ASSERT_TRUE(CIB);
  auto PreambleB = buildPreamble(getFullPath("b/Use.cpp"), *CIB, UseB,
                                 /*InMemory=*/true, /*Callback=*/nullptr);
  ASSERT_TRUE(PreambleB);
  auto ASTB = ParsedAST::build(getFullPath("b/Use.cpp"), UseB, std::move(CIB),
                               {}, PreambleB);
  ASSERT_TRUE(ASTB);
  EXPECT_TRUE(findDecl(*ASTB, "onlyB").isFromASTFile());
}

TEST_F(PrerequisiteModulesTests, PersistentModuleCacheReusedAcrossBuilders) {
  MockDirectoryCompilationDatabase CDB(TestDir, FS);

  CDB.addFile("M.cppm", R"cpp(
export module M;
export int MValue = 43;
  )cpp");
  CDB.addFile("A.cppm", R"cpp(
export module A;
import M;
export int AValue = MValue;
  )cpp");

  std::string FirstPCMPath;
  {
    ModulesBuilder Builder(CDB);
    auto AInfo = Builder.buildPrerequisiteModulesFor(getFullPath("A.cppm"), FS);
    ASSERT_TRUE(AInfo);
    HeaderSearchOptions HS(TestDir);
    AInfo->adjustHeaderSearchOptions(HS);
    ASSERT_EQ(HS.PrebuiltModuleFiles.count("M"), 1u);
    FirstPCMPath = HS.PrebuiltModuleFiles["M"];
    EXPECT_TRUE(llvm::sys::fs::exists(FirstPCMPath));
    EXPECT_TRUE(StringRef(FirstPCMPath).contains(".cache/clangd/modules"));
  }

  EXPECT_FALSE(llvm::sys::fs::exists(FirstPCMPath));

  // A fresh builder should reuse the persistent BMI published by the first one
  // instead of rebuilding its stable cache entry.
  ModulesBuilder Builder(CDB);
  auto AInfo = Builder.buildPrerequisiteModulesFor(getFullPath("A.cppm"), FS);
  ASSERT_TRUE(AInfo);
  HeaderSearchOptions HS(TestDir);
  AInfo->adjustHeaderSearchOptions(HS);
  ASSERT_EQ(HS.PrebuiltModuleFiles.count("M"), 1u);
  EXPECT_TRUE(llvm::sys::fs::exists(HS.PrebuiltModuleFiles["M"]));
  EXPECT_TRUE(StringRef(HS.PrebuiltModuleFiles["M"]).contains("M-"));

  ParseInputs AUse = getInputs("A.cppm", CDB);
  AUse.ModulesManager = &Builder;
  auto Invocation = buildCompilerInvocation(AUse, DiagConsumer);
  ASSERT_TRUE(Invocation);
  EXPECT_TRUE(AInfo->canReuse(*Invocation, FS.view(TestDir)));
}

TEST_F(PrerequisiteModulesTests,
       PersistentModuleCacheRebuildsAfterDeletingStalePCM) {
  MockDirectoryCompilationDatabase CDB(TestDir, FS);

  CDB.addFile("M.cppm", R"cpp(
export module M;
export int MValue = 43;
  )cpp");
  CDB.addFile("A.cppm", R"cpp(
export module A;
import M;
export int AValue = MValue;
  )cpp");

  ModulesBuilder Builder(CDB);
  auto AInfo = Builder.buildPrerequisiteModulesFor(getFullPath("A.cppm"), FS);
  ASSERT_TRUE(AInfo);
  HeaderSearchOptions HS(TestDir);
  AInfo->adjustHeaderSearchOptions(HS);
  ASSERT_EQ(HS.PrebuiltModuleFiles.count("M"), 1u);
  std::string PCMPath = HS.PrebuiltModuleFiles["M"];

  std::error_code EC;
  llvm::raw_fd_ostream OS(PCMPath, EC);
  ASSERT_FALSE(EC);
  OS << "broken";
  OS.close();

  // Corrupt the handed-out BMI and ensure clangd rebuilds a usable replacement.
  auto NewAInfo =
      Builder.buildPrerequisiteModulesFor(getFullPath("A.cppm"), FS);
  ASSERT_TRUE(NewAInfo);
  HeaderSearchOptions NewHS(TestDir);
  NewAInfo->adjustHeaderSearchOptions(NewHS);
  ASSERT_EQ(NewHS.PrebuiltModuleFiles.count("M"), 1u);
  EXPECT_TRUE(llvm::sys::fs::exists(NewHS.PrebuiltModuleFiles["M"]));
  EXPECT_TRUE(StringRef(NewHS.PrebuiltModuleFiles["M"]).contains("M-"));

  ParseInputs AUse = getInputs("A.cppm", CDB);
  AUse.ModulesManager = &Builder;
  auto Invocation = buildCompilerInvocation(AUse, DiagConsumer);
  ASSERT_TRUE(Invocation);
  EXPECT_TRUE(NewAInfo->canReuse(*Invocation, FS.view(TestDir)));
}

TEST_F(PrerequisiteModulesTests, PersistentModuleCacheCreatesSourceHashLock) {
  MockDirectoryCompilationDatabase CDB(TestDir, FS);

  CDB.addFile("M.cppm", R"cpp(
export module M;
export int MValue = 43;
  )cpp");
  CDB.addFile("A.cppm", R"cpp(
export module A;
import M;
export int AValue = MValue;
  )cpp");

  ModulesBuilder Builder(CDB);
  auto AInfo = Builder.buildPrerequisiteModulesFor(getFullPath("A.cppm"), FS);
  ASSERT_TRUE(AInfo);

  HeaderSearchOptions HS(TestDir);
  AInfo->adjustHeaderSearchOptions(HS);
  ASSERT_EQ(HS.PrebuiltModuleFiles.count("M"), 1u);

  llvm::SmallString<256> PCMPath(HS.PrebuiltModuleFiles["M"]);
  llvm::sys::path::remove_filename(PCMPath);
  llvm::SmallString<256> SourceHashDir(PCMPath);
  llvm::sys::path::remove_filename(SourceHashDir);
  llvm::SmallString<256> CacheRoot(SourceHashDir);
  llvm::sys::path::remove_filename(CacheRoot);

  // Locks live next to the persistent cache and are keyed by source-hash so
  // builders publishing the same module unit serialize with each other.
  llvm::StringRef SourceDirectoryName =
      llvm::sys::path::filename(SourceHashDir);
  // Split from the right because the readable basename may also contain '-'.
  llvm::StringRef SourceHash = SourceDirectoryName.rsplit('-').second;
  llvm::SmallString<256> LockPath(CacheRoot);
  llvm::sys::path::append(LockPath, ".locks", SourceHash);

  EXPECT_TRUE(llvm::sys::fs::exists(LockPath));
}

TEST_F(PrerequisiteModulesTests,
       PersistentModuleCacheGCRemovesOldStablePublishedModule) {
  PerFileModulesCompilationDatabase CDB(TestDir, FS);

  CDB.addFile("M.cppm", R"cpp(
export module M;
export int MValue = 43;
  )cpp");
  CDB.addFile("A.cppm", R"cpp(
export module A;
import M;
export int AValue = MValue;
  )cpp");

  llvm::SmallString<256> OrphanPCMPath;
  {
    ModulesBuilder Builder(CDB);
    auto AInfo = Builder.buildPrerequisiteModulesFor(getFullPath("A.cppm"), FS);
    ASSERT_TRUE(AInfo);
    HeaderSearchOptions HS(TestDir);
    AInfo->adjustHeaderSearchOptions(HS);
    ASSERT_EQ(HS.PrebuiltModuleFiles.count("M"), 1u);

    OrphanPCMPath = HS.PrebuiltModuleFiles["M"];
    llvm::sys::path::remove_filename(OrphanPCMPath);
    llvm::sys::path::append(OrphanPCMPath, "Orphan.pcm");

    std::error_code EC;
    llvm::raw_fd_ostream OS(OrphanPCMPath, EC);
    ASSERT_FALSE(EC);
    OS << "orphan";
    OS.close();
    EXPECT_TRUE(llvm::sys::fs::exists(OrphanPCMPath));

    int FD = -1;
    ASSERT_FALSE(llvm::sys::fs::openFileForWrite(OrphanPCMPath, FD,
                                                 llvm::sys::fs::CD_OpenExisting,
                                                 llvm::sys::fs::OF_None));
    auto CloseFD = llvm::scope_exit(
        [&] { llvm::sys::Process::SafelyCloseFileDescriptor(FD); });
    llvm::sys::TimePoint<> OldTime =
        std::chrono::system_clock::now() - std::chrono::hours(24 * 5);
    ASSERT_FALSE(llvm::sys::fs::setLastAccessAndModificationTime(FD, OldTime));
  }

  ModulesBuilder Builder(CDB);
  auto AInfo = Builder.buildPrerequisiteModulesFor(getFullPath("A.cppm"), FS);
  ASSERT_TRUE(AInfo);
  EXPECT_FALSE(llvm::sys::fs::exists(OrphanPCMPath));
}

TEST_F(PrerequisiteModulesTests,
       PersistentModuleCacheGCKeepsRecentStablePublishedModule) {
  PerFileModulesCompilationDatabase CDB(TestDir, FS);

  CDB.addFile("M.cppm", R"cpp(
export module M;
export int MValue = 43;
  )cpp");
  CDB.addFile("A.cppm", R"cpp(
export module A;
import M;
export int AValue = MValue;
  )cpp");

  llvm::SmallString<256> OrphanPCMPath;
  {
    ModulesBuilder Builder(CDB);
    auto AInfo = Builder.buildPrerequisiteModulesFor(getFullPath("A.cppm"), FS);
    ASSERT_TRUE(AInfo);
    HeaderSearchOptions HS(TestDir);
    AInfo->adjustHeaderSearchOptions(HS);
    ASSERT_EQ(HS.PrebuiltModuleFiles.count("M"), 1u);

    OrphanPCMPath = HS.PrebuiltModuleFiles["M"];
    llvm::sys::path::remove_filename(OrphanPCMPath);
    llvm::sys::path::append(OrphanPCMPath, "Orphan.pcm");

    std::error_code EC;
    llvm::raw_fd_ostream OS(OrphanPCMPath, EC);
    ASSERT_FALSE(EC);
    OS << "orphan";
    OS.close();
    EXPECT_TRUE(llvm::sys::fs::exists(OrphanPCMPath));
  }

  ModulesBuilder Builder(CDB);
  auto AInfo = Builder.buildPrerequisiteModulesFor(getFullPath("A.cppm"), FS);
  ASSERT_TRUE(AInfo);
  EXPECT_TRUE(llvm::sys::fs::exists(OrphanPCMPath));
}

TEST_F(PrerequisiteModulesTests,
       PersistentModuleCacheGCRemovesOldVersionedModuleFile) {
  PerFileModulesCompilationDatabase CDB(TestDir, FS);

  CDB.addFile("M.cppm", R"cpp(
export module M;
export int MValue = 43;
  )cpp");
  CDB.addFile("A.cppm", R"cpp(
export module A;
import M;
export int AValue = MValue;
  )cpp");

  llvm::SmallString<256> OldVersionedPCMPath;
  {
    ModulesBuilder Builder(CDB);
    auto AInfo = Builder.buildPrerequisiteModulesFor(getFullPath("A.cppm"), FS);
    ASSERT_TRUE(AInfo);
    HeaderSearchOptions HS(TestDir);
    AInfo->adjustHeaderSearchOptions(HS);
    ASSERT_EQ(HS.PrebuiltModuleFiles.count("M"), 1u);

    OldVersionedPCMPath = HS.PrebuiltModuleFiles["M"];
    ASSERT_TRUE(llvm::sys::fs::exists(OldVersionedPCMPath));

    int FD = -1;
    ASSERT_FALSE(llvm::sys::fs::openFileForWrite(OldVersionedPCMPath, FD,
                                                 llvm::sys::fs::CD_OpenExisting,
                                                 llvm::sys::fs::OF_None));
    auto CloseFD = llvm::scope_exit(
        [&] { llvm::sys::Process::SafelyCloseFileDescriptor(FD); });
    llvm::sys::TimePoint<> OldTime =
        std::chrono::system_clock::now() - std::chrono::hours(24 * 5);
    ASSERT_FALSE(llvm::sys::fs::setLastAccessAndModificationTime(FD, OldTime));
  }

  ModulesBuilder Builder(CDB);
  auto AInfo = Builder.buildPrerequisiteModulesFor(getFullPath("A.cppm"), FS);
  ASSERT_TRUE(AInfo);
  EXPECT_FALSE(llvm::sys::fs::exists(OldVersionedPCMPath));
}

TEST_F(PrerequisiteModulesTests,
       PersistentModuleCacheGCKeepsRecentVersionedModuleFile) {
  PerFileModulesCompilationDatabase CDB(TestDir, FS);

  CDB.addFile("M.cppm", R"cpp(
export module M;
export int MValue = 43;
  )cpp");
  CDB.addFile("A.cppm", R"cpp(
export module A;
import M;
export int AValue = MValue;
  )cpp");

  auto FirstBuilder = std::make_unique<ModulesBuilder>(CDB);
  auto AInfo =
      FirstBuilder->buildPrerequisiteModulesFor(getFullPath("A.cppm"), FS);
  ASSERT_TRUE(AInfo);
  HeaderSearchOptions HS(TestDir);
  AInfo->adjustHeaderSearchOptions(HS);
  ASSERT_EQ(HS.PrebuiltModuleFiles.count("M"), 1u);
  llvm::StringRef CopyOnReadPCMPath = HS.PrebuiltModuleFiles["M"];
  ASSERT_TRUE(llvm::sys::fs::exists(CopyOnReadPCMPath));

  ModulesBuilder SecondBuilder(CDB);
  auto SecondInfo =
      SecondBuilder.buildPrerequisiteModulesFor(getFullPath("A.cppm"), FS);
  ASSERT_TRUE(SecondInfo);
  EXPECT_TRUE(llvm::sys::fs::exists(CopyOnReadPCMPath));
}

TEST_F(PrerequisiteModulesTests,
       PersistentModuleCacheIgnoresRequiredSourceForOnDiskPath) {
  ModuleUnitRootCompilationDatabase CDB(TestDir, FS);

  CDB.addFile("shared/M.cppm", R"cpp(
export module M;
export int MValue = 43;
  )cpp");
  CDB.addFile("projA/A.cppm", R"cpp(
export module A;
import M;
export int AValue = MValue;
  )cpp");
  CDB.addFile("projB/B.cppm", R"cpp(
export module B;
import M;
export int BValue = MValue;
  )cpp");

  ModulesBuilder Builder(CDB);

  auto AInfo =
      Builder.buildPrerequisiteModulesFor(getFullPath("projA/A.cppm"), FS);
  auto BInfo =
      Builder.buildPrerequisiteModulesFor(getFullPath("projB/B.cppm"), FS);
  ASSERT_TRUE(AInfo);
  ASSERT_TRUE(BInfo);

  HeaderSearchOptions HSA(TestDir);
  HeaderSearchOptions HSB(TestDir);
  AInfo->adjustHeaderSearchOptions(HSA);
  BInfo->adjustHeaderSearchOptions(HSB);
  ASSERT_EQ(HSA.PrebuiltModuleFiles.count("M"), 1u);
  ASSERT_EQ(HSB.PrebuiltModuleFiles.count("M"), 1u);
  EXPECT_TRUE(llvm::sys::fs::exists(HSA.PrebuiltModuleFiles["M"]));
  EXPECT_TRUE(llvm::sys::fs::exists(HSB.PrebuiltModuleFiles["M"]));

  llvm::SmallString<256> ExpectedRoot(getFullPath("shared/M.cppm"));
  llvm::sys::path::remove_filename(ExpectedRoot);
  llvm::sys::path::append(ExpectedRoot, ".cache", "clangd", "modules");
  EXPECT_TRUE(
      StringRef(HSA.PrebuiltModuleFiles["M"]).starts_with(ExpectedRoot));
  EXPECT_TRUE(StringRef(HSA.PrebuiltModuleFiles["M"]).contains("M.cppm-"));
  EXPECT_TRUE(
      StringRef(HSB.PrebuiltModuleFiles["M"]).starts_with(ExpectedRoot));
  EXPECT_TRUE(StringRef(HSB.PrebuiltModuleFiles["M"]).contains("M.cppm-"));
}

TEST_F(PrerequisiteModulesTests, ModuleImportThroughInclude) {
  MockDirectoryCompilationDatabase CDB(TestDir, FS);

  Annotations UseCpp(R"cpp(
#include "Header.hpp"
void use() {
  TypeFrom^Module t1;
  TypeFromHeader t2;
}
)cpp");

  CDB.addFile("M.cppm", R"cpp(
export module M;
export struct TypeFromModule {};
)cpp");

  CDB.addFile("Header.hpp", R"cpp(
import M;
struct TypeFromHeader {};
)cpp");

  CDB.addFile("Use.cpp", UseCpp.code());

  ModulesBuilder Builder(CDB);

  auto Inputs = getInputs("Use.cpp", CDB);
  Inputs.ModulesManager = &Builder;
  Inputs.Opts.SkipPreambleBuild = true;

  auto CI = buildCompilerInvocation(Inputs, DiagConsumer);
  ASSERT_TRUE(CI);

  auto Preamble =
      buildPreamble(getFullPath("Use.cpp"), *CI, Inputs, /*StoreInMemory=*/true,
                    /*PeambleCallback=*/nullptr);
  ASSERT_TRUE(Preamble);
  EXPECT_EQ(Preamble->Preamble.getBounds().Size, 0u);

  auto AST = ParsedAST::build(getFullPath("Use.cpp"), Inputs, std::move(CI), {},
                              Preamble);
  ASSERT_TRUE(AST);
  EXPECT_TRUE(AST->getDiagnostics().empty());

  auto Result = codeComplete(getFullPath("Use.cpp"), UseCpp.point(),
                             Preamble.get(), Inputs, {});
  EXPECT_THAT(Result.Completions,
              testing::UnorderedElementsAre(named("TypeFromModule"),
                                            named("TypeFromHeader")));
}

} // namespace
} // namespace clang::clangd

#endif
