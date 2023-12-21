//===-- StdLibTests.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "ClangdServer.h"
#include "CodeComplete.h"
#include "Compiler.h"
#include "Config.h"
#include "SyncAPI.h"
#include "TestFS.h"
#include "index/StdLib.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>

using namespace testing;

namespace clang {
namespace clangd {
namespace {

// Check the generated header sources contains usual standard library headers.
TEST(StdLibTests, getStdlibUmbrellaHeader) {
  LangOptions LO;
  LO.CPlusPlus = true;

  auto CXX = getStdlibUmbrellaHeader(LO).str();
  EXPECT_THAT(CXX, HasSubstr("#include <string>"));
  EXPECT_THAT(CXX, HasSubstr("#include <cstdio>"));
  EXPECT_THAT(CXX, Not(HasSubstr("#include <ios646.h>")));

  LO.CPlusPlus = false;
  auto C = getStdlibUmbrellaHeader(LO).str();
  EXPECT_THAT(C, Not(HasSubstr("#include <string>")));
  EXPECT_THAT(C, Not(HasSubstr("#include <cstdio>")));
  EXPECT_THAT(C, HasSubstr("#include <stdio.h>"));
}

MATCHER_P(Named, Name, "") { return arg.Name == Name; }

// Build an index, and check if it contains the right symbols.
TEST(StdLibTests, indexStandardLibrary) {
  MockFS FS;
  FS.Files["std/foo.h"] = R"cpp(
  #include <platform_stuff.h>
  #if __cplusplus >= 201703L
    int foo17();
  #elif __cplusplus >= 201402L
    int foo14();
  #else
    bool foo98();
  #endif
  )cpp";
  FS.Files["nonstd/platform_stuff.h"] = "int magic = 42;";

  ParseInputs OriginalInputs;
  OriginalInputs.TFS = &FS;
  OriginalInputs.CompileCommand.Filename = testPath("main.cc");
  OriginalInputs.CompileCommand.CommandLine = {"clang++", testPath("main.cc"),
                                               "-isystemstd/",
                                               "-isystemnonstd/", "-std=c++14"};
  OriginalInputs.CompileCommand.Directory = testRoot();
  IgnoreDiagnostics Diags;
  auto CI = buildCompilerInvocation(OriginalInputs, Diags);
  ASSERT_TRUE(CI);

  StdLibLocation Loc;
  Loc.Paths.push_back(testPath("std/"));

  auto Symbols =
      indexStandardLibrary("#include <foo.h>", std::move(CI), Loc, FS);
  EXPECT_THAT(Symbols, ElementsAre(Named("foo14")));
}

TEST(StdLibTests, StdLibSet) {
  StdLibSet Set;
  MockFS FS;
  FS.Files["std/_"] = "";
  FS.Files["libc/_"] = "";

  auto Add = [&](const LangOptions &LO,
                 std::vector<llvm::StringRef> SearchPath) {
    SourceManagerForFile SM("scratch", "");
    SM.get().getFileManager().setVirtualFileSystem(FS.view(std::nullopt));
    HeaderSearch HS(/*HSOpts=*/nullptr, SM.get(), SM.get().getDiagnostics(), LO,
                    /*Target=*/nullptr);
    for (auto P : SearchPath)
      HS.AddSearchPath(
          DirectoryLookup(
              cantFail(SM.get().getFileManager().getDirectoryRef(testPath(P))),
              SrcMgr::C_System, /*isFramework=*/false),
          true);
    return Set.add(LO, HS);
  };

  Config Cfg;
  Cfg.Index.StandardLibrary = false;
  WithContextValue Disabled(Config::Key, std::move(Cfg));

  LangOptions LO;
  LO.CPlusPlus = true;
  EXPECT_FALSE(Add(LO, {"std"})) << "Disabled in config";

  Cfg = Config();
  Cfg.Index.StandardLibrary = true;
  WithContextValue Enabled(Config::Key, std::move(Cfg));

  EXPECT_FALSE(Add(LO, {"std"})) << "No <vector> found";
  FS.Files["std/vector"] = "class vector;";
  EXPECT_TRUE(Add(LO, {"std"})) << "Indexing as C++98";
  EXPECT_FALSE(Add(LO, {"std"})) << "Don't reindex";
  LO.CPlusPlus11 = true;
  EXPECT_TRUE(Add(LO, {"std"})) << "Indexing as C++11";
  LO.CPlusPlus = false;
  EXPECT_FALSE(Add(LO, {"libc"})) << "No <stdio.h>";
  FS.Files["libc/stdio.h"] = true;
  EXPECT_TRUE(Add(LO, {"libc"})) << "Indexing as C";
}

MATCHER_P(StdlibSymbol, Name, "") {
  return arg.Name == Name && arg.Includes.size() == 1 &&
         llvm::StringRef(arg.Includes.front().Header).starts_with("<");
}

TEST(StdLibTests, EndToEnd) {
  Config Cfg;
  Cfg.Index.StandardLibrary = true;
  WithContextValue Enabled(Config::Key, std::move(Cfg));

  MockFS FS;
  FS.Files["stdlib/vector"] =
      "namespace std { template <class> class vector; }";
  FS.Files["stdlib/list"] =
      " namespace std { template <typename T> class list; }";
  MockCompilationDatabase CDB;
  CDB.ExtraClangFlags.push_back("-isystem" + testPath("stdlib"));
  ClangdServer::Options Opts = ClangdServer::optsForTest();
  Opts.BuildDynamicSymbolIndex = true; // also used for stdlib index
  ClangdServer Server(CDB, FS, Opts);

  Annotations A("std::^");

  Server.addDocument(testPath("foo.cc"), A.code());
  ASSERT_TRUE(Server.blockUntilIdleForTest());
  clangd::CodeCompleteOptions CCOpts;
  auto Completions =
      cantFail(runCodeComplete(Server, testPath("foo.cc"), A.point(), CCOpts));
  EXPECT_THAT(
      Completions.Completions,
      UnorderedElementsAre(StdlibSymbol("list"), StdlibSymbol("vector")));
}

} // namespace
} // namespace clangd
} // namespace clang
