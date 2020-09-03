//===--- HeaderSourceSwitchTests.cpp - ---------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HeaderSourceSwitch.h"

#include "SyncAPI.h"
#include "TestFS.h"
#include "TestTU.h"
#include "index/MemIndex.h"
#include "llvm/ADT/None.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

TEST(HeaderSourceSwitchTest, FileHeuristic) {
  MockFS FS;
  auto FooCpp = testPath("foo.cpp");
  auto FooH = testPath("foo.h");
  auto Invalid = testPath("main.cpp");

  FS.Files[FooCpp];
  FS.Files[FooH];
  FS.Files[Invalid];
  Optional<Path> PathResult =
      getCorrespondingHeaderOrSource(FooCpp, FS.view(llvm::None));
  EXPECT_TRUE(PathResult.hasValue());
  ASSERT_EQ(PathResult.getValue(), FooH);

  PathResult = getCorrespondingHeaderOrSource(FooH, FS.view(llvm::None));
  EXPECT_TRUE(PathResult.hasValue());
  ASSERT_EQ(PathResult.getValue(), FooCpp);

  // Test with header file in capital letters and different extension, source
  // file with different extension
  auto FooC = testPath("bar.c");
  auto FooHH = testPath("bar.HH");

  FS.Files[FooC];
  FS.Files[FooHH];
  PathResult = getCorrespondingHeaderOrSource(FooC, FS.view(llvm::None));
  EXPECT_TRUE(PathResult.hasValue());
  ASSERT_EQ(PathResult.getValue(), FooHH);

  // Test with both capital letters
  auto Foo2C = testPath("foo2.C");
  auto Foo2HH = testPath("foo2.HH");
  FS.Files[Foo2C];
  FS.Files[Foo2HH];
  PathResult = getCorrespondingHeaderOrSource(Foo2C, FS.view(llvm::None));
  EXPECT_TRUE(PathResult.hasValue());
  ASSERT_EQ(PathResult.getValue(), Foo2HH);

  // Test with source file as capital letter and .hxx header file
  auto Foo3C = testPath("foo3.C");
  auto Foo3HXX = testPath("foo3.hxx");

  FS.Files[Foo3C];
  FS.Files[Foo3HXX];
  PathResult = getCorrespondingHeaderOrSource(Foo3C, FS.view(llvm::None));
  EXPECT_TRUE(PathResult.hasValue());
  ASSERT_EQ(PathResult.getValue(), Foo3HXX);

  // Test if asking for a corresponding file that doesn't exist returns an empty
  // string.
  PathResult = getCorrespondingHeaderOrSource(Invalid, FS.view(llvm::None));
  EXPECT_FALSE(PathResult.hasValue());
}

MATCHER_P(DeclNamed, Name, "") {
  if (const NamedDecl *ND = dyn_cast<NamedDecl>(arg))
    if (ND->getQualifiedNameAsString() == Name)
      return true;
  return false;
}

TEST(HeaderSourceSwitchTest, GetLocalDecls) {
  TestTU TU;
  TU.HeaderCode = R"cpp(
  void HeaderOnly();
  )cpp";
  TU.Code = R"cpp(
  void MainF1();
  class Foo {};
  namespace ns {
  class Foo {
    void method();
    int field;
  };
  } // namespace ns

  // Non-indexable symbols
  namespace {
  void Ignore1() {}
  }

  )cpp";

  auto AST = TU.build();
  EXPECT_THAT(getIndexableLocalDecls(AST),
              testing::UnorderedElementsAre(
                  DeclNamed("MainF1"), DeclNamed("Foo"), DeclNamed("ns::Foo"),
                  DeclNamed("ns::Foo::method"), DeclNamed("ns::Foo::field")));
}

TEST(HeaderSourceSwitchTest, FromHeaderToSource) {
  // build a proper index, which contains symbols:
  //   A_Sym1, declared in TestTU.h, defined in a.cpp
  //   B_Sym[1-2], declared in TestTU.h, defined in b.cpp
  SymbolSlab::Builder AllSymbols;
  TestTU Testing;
  Testing.HeaderFilename = "TestTU.h";
  Testing.HeaderCode = "void A_Sym1();";
  Testing.Filename = "a.cpp";
  Testing.Code = "void A_Sym1() {};";
  for (auto &Sym : Testing.headerSymbols())
    AllSymbols.insert(Sym);

  Testing.HeaderCode = R"cpp(
  void B_Sym1();
  void B_Sym2();
  void B_Sym3_NoDef();
  )cpp";
  Testing.Filename = "b.cpp";
  Testing.Code = R"cpp(
  void B_Sym1() {}
  void B_Sym2() {}
  )cpp";
  for (auto &Sym : Testing.headerSymbols())
    AllSymbols.insert(Sym);
  auto Index = MemIndex::build(std::move(AllSymbols).build(), {}, {});

  // Test for switch from .h header to .cc source
  struct {
    llvm::StringRef HeaderCode;
    llvm::Optional<std::string> ExpectedSource;
  } TestCases[] = {
      {"// empty, no header found", llvm::None},
      {R"cpp(
         // no definition found in the index.
         void NonDefinition();
       )cpp",
       llvm::None},
      {R"cpp(
         void A_Sym1();
       )cpp",
       testPath("a.cpp")},
      {R"cpp(
         // b.cpp wins.
         void A_Sym1();
         void B_Sym1();
         void B_Sym2();
       )cpp",
       testPath("b.cpp")},
      {R"cpp(
         // a.cpp and b.cpp have same scope, but a.cpp because "a.cpp" < "b.cpp".
         void A_Sym1();
         void B_Sym1();
       )cpp",
       testPath("a.cpp")},

       {R"cpp(
          // We don't have definition in the index, so stay in the header.
          void B_Sym3_NoDef();
       )cpp",
       None},
  };
  for (const auto &Case : TestCases) {
    TestTU TU = TestTU::withCode(Case.HeaderCode);
    TU.Filename = "TestTU.h";
    TU.ExtraArgs.push_back("-xc++-header"); // inform clang this is a header.
    auto HeaderAST = TU.build();
    EXPECT_EQ(Case.ExpectedSource,
              getCorrespondingHeaderOrSource(testPath(TU.Filename), HeaderAST,
                                             Index.get()));
  }
}

TEST(HeaderSourceSwitchTest, FromSourceToHeader) {
  // build a proper index, which contains symbols:
  //   A_Sym1, declared in a.h, defined in TestTU.cpp
  //   B_Sym[1-2], declared in b.h, defined in TestTU.cpp
  TestTU TUForIndex = TestTU::withCode(R"cpp(
  #include "a.h"
  #include "b.h"

  void A_Sym1() {}

  void B_Sym1() {}
  void B_Sym2() {}
  )cpp");
  TUForIndex.AdditionalFiles["a.h"] = R"cpp(
  void A_Sym1();
  )cpp";
  TUForIndex.AdditionalFiles["b.h"] = R"cpp(
  void B_Sym1();
  void B_Sym2();
  )cpp";
  TUForIndex.Filename = "TestTU.cpp";
  auto Index = TUForIndex.index();

  // Test for switching from .cc source file to .h header.
  struct {
    llvm::StringRef SourceCode;
    llvm::Optional<std::string> ExpectedResult;
  } TestCases[] = {
      {"// empty, no header found", llvm::None},
      {R"cpp(
         // symbol not in index, no header found
         void Local() {}
       )cpp",
       llvm::None},

      {R"cpp(
         // a.h wins.
         void A_Sym1() {}
       )cpp",
       testPath("a.h")},

      {R"cpp(
         // b.h wins.
         void A_Sym1() {}
         void B_Sym1() {}
         void B_Sym2() {}
       )cpp",
       testPath("b.h")},

      {R"cpp(
         // a.h and b.h have same scope, but a.h wins because "a.h" < "b.h".
         void A_Sym1() {}
         void B_Sym1() {}
       )cpp",
       testPath("a.h")},
  };
  for (const auto &Case : TestCases) {
    TestTU TU = TestTU::withCode(Case.SourceCode);
    TU.Filename = "Test.cpp";
    auto AST = TU.build();
    EXPECT_EQ(Case.ExpectedResult,
              getCorrespondingHeaderOrSource(testPath(TU.Filename), AST,
                                             Index.get()));
  }
}

TEST(HeaderSourceSwitchTest, ClangdServerIntegration) {
  MockCompilationDatabase CDB;
  CDB.ExtraClangFlags = {"-I" +
                         testPath("src/include")}; // add search directory.
  MockFS FS;
  // File heuristic fails here, we rely on the index to find the .h file.
  std::string CppPath = testPath("src/lib/test.cpp");
  std::string HeaderPath = testPath("src/include/test.h");
  FS.Files[HeaderPath] = "void foo();";
  const std::string FileContent = R"cpp(
    #include "test.h"
    void foo() {};
  )cpp";
  FS.Files[CppPath] = FileContent;
  auto Options = ClangdServer::optsForTest();
  Options.BuildDynamicSymbolIndex = true;
  ClangdServer Server(CDB, FS, Options);
  runAddDocument(Server, CppPath, FileContent);
  EXPECT_EQ(HeaderPath,
            *llvm::cantFail(runSwitchHeaderSource(Server, CppPath)));
}

} // namespace
} // namespace clangd
} // namespace clang
