//===------------ ModuleDependencyScannerTest.cpp ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// FIXME: Skip testing on windows temporarily due to the different escaping
/// code mode.
#ifndef _WIN32

#include "ModuleDependencyScanner.h"
#include "ModulesTestSetup.h"
#include "TestFS.h"

using namespace clang;
using namespace clang::clangd;
using namespace clang::tooling::dependencies;

namespace {
class ModuleDependencyScannerTests : public ModuleTestSetup {};

TEST_F(ModuleDependencyScannerTests, SingleFile) {
  addFile("foo.h", R"cpp(
import foo;
  )cpp");

  addFile("A.cppm", R"cpp(
module;
#include "foo.h"
export module A;
export import :partA;
import :partB;
import C;

module :private;
import D;
  )cpp");

  MockCompilationDatabase CDB(TestDir);
  CDB.ExtraClangFlags.push_back("-std=c++20");

  ModuleDependencyScanner Scanner(CDB, TFS);
  std::optional<ModuleDependencyScanner::ModuleDependencyInfo> ScanningResult =
      Scanner.scan(getFullPath("A.cppm"));
  EXPECT_TRUE(ScanningResult);

  EXPECT_TRUE(ScanningResult->ModuleName);
  EXPECT_EQ(*ScanningResult->ModuleName, "A");

  EXPECT_EQ(ScanningResult->RequiredModules.size(), 5u);
  EXPECT_EQ(ScanningResult->RequiredModules[0], "foo");
  EXPECT_EQ(ScanningResult->RequiredModules[1], "A:partA");
  EXPECT_EQ(ScanningResult->RequiredModules[2], "A:partB");
  EXPECT_EQ(ScanningResult->RequiredModules[3], "C");
  EXPECT_EQ(ScanningResult->RequiredModules[4], "D");
}

TEST_F(ModuleDependencyScannerTests, GlobalScanning) {
  addFile("build/compile_commands.json", R"cpp(
[
{
  "directory": "__DIR__",
  "command": "clang++ -std=c++20 __DIR__/foo.cppm -fmodule-output=__DIR__/foo.pcm -c -o __DIR__/foo.o",
  "file": "__DIR__/foo.cppm",
  "output": "__DIR__/foo.o"
},
{
  "directory": "__DIR__",
  "command": "clang++ -std=c++20 __DIR__/C.cppm -fmodule-output=__DIR__/C.pcm -c -o __DIR__/C.o",
  "file": "__DIR__/C.cppm",
  "output": "__DIR__/C.o"
},
{
  "directory": "__DIR__",
  "command": "clang++ -std=c++20 __DIR__/D.cppm -fmodule-output=__DIR__/D.pcm -c -o __DIR__/D.o",
  "file": "__DIR__/D.cppm",
  "output": "__DIR__/D.o"
},
{
  "directory": "__DIR__",
  "command": "clang++ -std=c++20 __DIR__/A-partA.cppm -fmodule-file=foo=__DIR__/foo.pcm -fmodule-output=__DIR__/A-partA.pcm -c -o __DIR__/A-partA.o",
  "file": "__DIR__/A-partA.cppm",
  "output": "__DIR__/A-partA.o"
},
{
  "directory": "__DIR__",
  "command": "clang++ -std=c++20 __DIR__/A-partB.cppm -fmodule-file=C=__DIR__/C.pcm -fmodule-output=__DIR__/A-partB.pcm -c -o __DIR__/A-partB.o",
  "file": "__DIR__/A-partB.cppm",
  "output": "__DIR__/A-partB.o"
},
{
  "directory": "__DIR__",
  "command": "clang++ -std=c++20 __DIR__/A.cppm -fmodule-file=A:partB=__DIR__/A-partB.pcm -fmodule-file=A:partA=__DIR__/A-partA.pcm -fmodule-file=foo=__DIR__/foo.pcm -fmodule-file=C=__DIR__/C.pcm -fmodule-file=D=__DIR__/C.pcm -fmodule-output=__DIR__/A.pcm -c -o __DIR__/A.o",
  "file": "__DIR__/A.cppm",
  "output": "__DIR__/A.o"
},
]
  )cpp");

  addFile("foo.cppm", R"cpp(
export module foo;
  )cpp");

  addFile("foo.h", R"cpp(
import foo;
  )cpp");

  addFile("A-partA.cppm", R"cpp(
export module A:partA;
import foo;
  )cpp");

  addFile("A-partB.cppm", R"cpp(
module A:partB;
import C;
  )cpp");

  addFile("C.cppm", R"cpp(
export module C;
  )cpp");

  addFile("D.cppm", R"cpp(
export module D;
  )cpp");

  addFile("A.cppm", R"cpp(
module;
#include "foo.h"
export module A;
export import :partA;
import :partB;
import C;

module :private;
import D;
  )cpp");

  std::unique_ptr<GlobalCompilationDatabase> CDB =
      getGlobalCompilationDatabase();
  ModuleDependencyScanner Scanner(*CDB.get(), TFS);
  Scanner.globalScan({getFullPath("A.cppm"), getFullPath("foo.cppm"),
                      getFullPath("A-partA.cppm"), getFullPath("A-partB.cppm"),
                      getFullPath("C.cppm"), getFullPath("D.cppm")});

  EXPECT_TRUE(Scanner.getSourceForModuleName("foo").endswith("foo.cppm"));
  EXPECT_TRUE(Scanner.getSourceForModuleName("A").endswith("A.cppm"));
  EXPECT_TRUE(
      Scanner.getSourceForModuleName("A:partA").endswith("A-partA.cppm"));
  EXPECT_TRUE(
      Scanner.getSourceForModuleName("A:partB").endswith("A-partB.cppm"));
  EXPECT_TRUE(Scanner.getSourceForModuleName("C").endswith("C.cppm"));
  EXPECT_TRUE(Scanner.getSourceForModuleName("D").endswith("D.cppm"));

  EXPECT_TRUE(Scanner.getRequiredModules(getFullPath("foo.cppm")).empty());
  EXPECT_TRUE(Scanner.getRequiredModules(getFullPath("C.cppm")).empty());
  EXPECT_TRUE(Scanner.getRequiredModules(getFullPath("D.cppm")).empty());

  EXPECT_EQ(Scanner.getRequiredModules(getFullPath("A-partA.cppm")).size(), 1u);
  EXPECT_EQ(Scanner.getRequiredModules(getFullPath("A-partA.cppm"))[0], "foo");

  EXPECT_EQ(Scanner.getRequiredModules(getFullPath("A-partB.cppm")).size(), 1u);
  EXPECT_EQ(Scanner.getRequiredModules(getFullPath("A-partB.cppm"))[0], "C");

  EXPECT_EQ(Scanner.getRequiredModules(getFullPath("A.cppm")).size(), 5u);
  EXPECT_EQ(Scanner.getRequiredModules(getFullPath("A.cppm"))[0], "foo");
  EXPECT_EQ(Scanner.getRequiredModules(getFullPath("A.cppm"))[1], "A:partA");
  EXPECT_EQ(Scanner.getRequiredModules(getFullPath("A.cppm"))[2], "A:partB");
  EXPECT_EQ(Scanner.getRequiredModules(getFullPath("A.cppm"))[3], "C");
  EXPECT_EQ(Scanner.getRequiredModules(getFullPath("A.cppm"))[4], "D");
}

} // namespace

#endif
