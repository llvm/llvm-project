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
#include "ModulesTestSetup.h"

using namespace clang;
using namespace clang::clangd;

namespace {
class PrerequisiteModulesTests : public ModuleTestSetup {};

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
      Builder.buildPrerequisiteModulesFor(getFullPath("NonModular.cpp"), &TFS);
  EXPECT_FALSE(NonModularInfo);

  auto MInfo = Builder.buildPrerequisiteModulesFor(getFullPath("M.cppm"), &TFS);
  // buildPrerequisiteModulesFor won't built the module itself.
  EXPECT_FALSE(MInfo);

  // Module N shouldn't be able to be built.
  auto NInfo = Builder.buildPrerequisiteModulesFor(getFullPath("N.cppm"), &TFS);
  EXPECT_TRUE(NInfo);
  EXPECT_TRUE(NInfo->isModuleUnitBuilt("M"));
  EXPECT_TRUE(NInfo->isModuleUnitBuilt("N:Part"));

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

    NInfo = Builder.buildPrerequisiteModulesFor(getFullPath("N.cppm"), &TFS);
    EXPECT_TRUE(NInfo && NInfo->canReuse(*Invocation, TFS.view(TestDir)));

    addFile("foo.h", R"cpp(
inline void foo() {}
inline void foo(int) {}
  )cpp");
    EXPECT_FALSE(NInfo && NInfo->canReuse(*Invocation, TFS.view(TestDir)));

    NInfo = Builder.buildPrerequisiteModulesFor(getFullPath("N.cppm"), &TFS);
    EXPECT_TRUE(NInfo && NInfo->canReuse(*Invocation, TFS.view(TestDir)));
  }

  addFile("N-part.cppm", R"cpp(
export module N:Part;
// Intentioned to make it uncompilable.
export int NPart = 4LIdjwldijaw
  )cpp");
  EXPECT_FALSE(NInfo && NInfo->canReuse(*Invocation, TFS.view(TestDir)));
  NInfo = Builder.buildPrerequisiteModulesFor(getFullPath("N.cppm"), &TFS);
  EXPECT_TRUE(NInfo);
  // So NInfo should be unreusable even after rebuild.
  EXPECT_FALSE(NInfo->canReuse(*Invocation, TFS.view(TestDir)));

  addFile("N-part.cppm", R"cpp(
export module N:Part;
export int NPart = 43;
  )cpp");
  EXPECT_TRUE(NInfo);
  EXPECT_FALSE(NInfo->canReuse(*Invocation, TFS.view(TestDir)));
  NInfo = Builder.buildPrerequisiteModulesFor(getFullPath("N.cppm"), &TFS);
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

} // namespace

#endif
