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
class PrerequisiteModulesTests : public ::testing::Test {
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

TEST_F(PrerequisiteModulesTests, PrerequisiteModulesTest) {
  addFile("build/compile_commands.json", R"cpp(
[
{
  "directory": "__DIR__",
  "command": "clang++ -std=c++20 __DIR__/M.cppm -fmodule-output=__DIR__/M.pcm -c -o __DIR__/M.o",
  "file": "__DIR__/M.cppm",
  "output": "__DIR__/M.o"
},
{
  "directory": "__DIR__",
  "command": "clang++ -std=c++20 __DIR__/N.cppm -fmodule-file=M=__DIR__/M.pcm -fmodule-file=N:Part=__DIR__/N-partition.pcm -fprebuilt-module-path=__DIR__ -fmodule-output=__DIR__/N.pcm -c -o __DIR__/N.o",
  "file": "__DIR__/N.cppm",
  "output": "__DIR__/N.o"
},
{
  "directory": "__DIR__",
  "command": "clang++ -std=c++20 __DIR__/N-part.cppm -fmodule-output=__DIR__/N-partition.pcm -c -o __DIR__/N-part.o",
  "file": "__DIR__/N-part.cppm",
  "output": "__DIR__/N-part.o"
},
{
  "directory": "__DIR__",
  "command": "clang++ -std=c++20 __DIR__/L.cppm -fmodule-output=__DIR__/L.pcm -c -o __DIR__/L.o",
  "file": "__DIR__/L.cppm",
  "output": "__DIR__/L.o"
},
{
  "directory": "__DIR__",
  "command": "clang++ -std=c++20 __DIR__/NonModular.cpp -c -o __DIR__/NonModular.o",
  "file": "__DIR__/NonModular.cpp",
  "output": "__DIR__/NonModular.o"
}
]
  )cpp");

  addFile("foo.h", R"cpp(
inline void foo() {}
  )cpp");

  addFile("M.cppm", R"cpp(
module;
#include "foo.h"
export module M;
  )cpp");

  addFile("N.cppm", R"cpp(
export module N;
import :Part;
import M;
  )cpp");

  addFile("N-part.cppm", R"cpp(
// Different name with filename intentionally.
export module N:Part;
  )cpp");

  addFile("bar.h", R"cpp(
inline void bar() {}
  )cpp");

  addFile("L.cppm", R"cpp(
module;
#include "bar.h"
export module L;
  )cpp");

  addFile("NonModular.cpp", R"cpp(
#include "bar.h"
#include "foo.h"
void use() {
  foo();
  bar();
}
  )cpp");

  std::unique_ptr<GlobalCompilationDatabase> CDB =
      getGlobalCompilationDatabase();
  EXPECT_TRUE(CDB);
  ModulesBuilder Builder(*CDB.get());

  // NonModular.cpp is not related to modules. So nothing should be built.
  auto NonModularInfo =
      Builder.buildPrerequisiteModulesFor(getFullPath("NonModular.cpp"), TFS);
  EXPECT_FALSE(NonModularInfo);

  auto MInfo = Builder.buildPrerequisiteModulesFor(getFullPath("M.cppm"), TFS);
  // buildPrerequisiteModulesFor won't built the module itself.
  EXPECT_FALSE(MInfo);

  // Module N shouldn't be able to be built.
  auto NInfo = Builder.buildPrerequisiteModulesFor(getFullPath("N.cppm"), TFS);
  EXPECT_TRUE(NInfo);

  ParseInputs NInput = getInputs("N.cppm", *CDB);
  std::vector<std::string> CC1Args;
  std::unique_ptr<CompilerInvocation> Invocation =
      getCompilerInvocation(NInput);
  // Test that `PrerequisiteModules::canReuse` works basically.
  EXPECT_TRUE(NInfo->canReuse(*Invocation, TFS.view(TestDir)));

  // Test that we can still reuse the NInfo after we touch a unrelated file.
  {
    addFile("L.cppm", R"cpp(
module;
#include "bar.h"
export module L;
export int ll = 43;
  )cpp");
    EXPECT_TRUE(NInfo->canReuse(*Invocation, TFS.view(TestDir)));

    addFile("bar.h", R"cpp(
inline void bar() {}
inline void bar(int) {}
  )cpp");
    EXPECT_TRUE(NInfo->canReuse(*Invocation, TFS.view(TestDir)));
  }

  // Test that we can't reuse the NInfo after we touch a related file.
  {
    addFile("M.cppm", R"cpp(
module;
#include "foo.h"
export module M;
export int mm = 44;
  )cpp");
    EXPECT_FALSE(NInfo && NInfo->canReuse(*Invocation, TFS.view(TestDir)));

    NInfo = Builder.buildPrerequisiteModulesFor(getFullPath("N.cppm"), TFS);
    EXPECT_TRUE(NInfo && NInfo->canReuse(*Invocation, TFS.view(TestDir)));

    addFile("foo.h", R"cpp(
inline void foo() {}
inline void foo(int) {}
  )cpp");
    EXPECT_FALSE(NInfo && NInfo->canReuse(*Invocation, TFS.view(TestDir)));

    NInfo = Builder.buildPrerequisiteModulesFor(getFullPath("N.cppm"), TFS);
    EXPECT_TRUE(NInfo && NInfo->canReuse(*Invocation, TFS.view(TestDir)));
  }

  addFile("N-part.cppm", R"cpp(
export module N:Part;
// Intentioned to make it uncompilable.
export int NPart = 4LIdjwldijaw
  )cpp");
  EXPECT_FALSE(NInfo && NInfo->canReuse(*Invocation, TFS.view(TestDir)));
  NInfo = Builder.buildPrerequisiteModulesFor(getFullPath("N.cppm"), TFS);
  EXPECT_TRUE(NInfo);
  // So NInfo should be unreusable even after rebuild.
  EXPECT_FALSE(NInfo->canReuse(*Invocation, TFS.view(TestDir)));

  addFile("N-part.cppm", R"cpp(
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
  addFile("N-part.cppm", R"cpp(
export module N:Part;
export int NPart = 43;
  )cpp");
  EXPECT_TRUE(NInfo && NInfo->canReuse(*Invocation, TFS.view(TestDir)));

  addFile("N.cppm", R"cpp(
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
    ParseInputs NInput = getInputs("N.cppm", *CDB);
    std::vector<std::string> CC1Args;
    std::unique_ptr<CompilerInvocation> NInvocation =
        getCompilerInvocation(NInput);
    HeaderSearchOptions &HSOpts = NInvocation->getHeaderSearchOpts();
    NInfo->adjustHeaderSearchOptions(HSOpts);

    EXPECT_TRUE(HSOpts.PrebuiltModuleFiles.count("M"));
    EXPECT_TRUE(HSOpts.PrebuiltModuleFiles.count("N:Part"));
  }
}

// An End-to-End test for modules.
TEST_F(PrerequisiteModulesTests, ParsedASTTest) {
  addFile("A.cppm", R"cpp(
export module A;
export void printA();
  )cpp");

  addFile("Use.cpp", R"cpp(
import A;
)cpp");

  addFile("build/compile_commands.json", R"cpp(
[
{
  "directory": "__DIR__",
  "command": "clang++ -std=c++20 __DIR__/A.cppm -fmodule-output=__DIR__/A.pcm -c -o __DIR__/A.o",
  "file": "__DIR__/A.cppm",
  "output": "__DIR__/A.o"
},
{
  "directory": "__DIR__",
  "command": "clang++ -std=c++20 __DIR__/Use.cpp -c -o __DIR__/Use.o",
  "file": "__DIR__/Use.cpp",
  "output": "__DIR__/Use.o"
}
]
  )cpp");

  std::unique_ptr<GlobalCompilationDatabase> CDB =
      getGlobalCompilationDatabase();
  EXPECT_TRUE(CDB);
  ModulesBuilder Builder(*CDB.get());

  ParseInputs Use = getInputs("Use.cpp", *CDB);
  Use.ModulesManager = &Builder;

  std::unique_ptr<CompilerInvocation> CI = getCompilerInvocation(Use);
  EXPECT_TRUE(CI);

  auto Preamble = buildPreamble(getFullPath("Use.cpp"), *CI, Use, /*InMemory=*/true,
      /*Callback=*/nullptr);
  EXPECT_TRUE(Preamble);
  EXPECT_TRUE(Preamble->RequiredModules);

  auto AST = ParsedAST::build(getFullPath("Use.cpp"), Use,
            std::move(CI), {}, Preamble);
  EXPECT_TRUE(AST);

  const NamedDecl &D = findDecl(*AST, "printA");
  EXPECT_TRUE(D.isFromASTFile());
}

} // namespace
} // namespace clang::clangd

#endif
