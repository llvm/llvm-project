//===-- FileIndexTests.cpp  ---------------------------*- C++ -*-----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AST.h"
#include "Annotations.h"
#include "ClangdUnit.h"
#include "SyncAPI.h"
#include "TestFS.h"
#include "TestTU.h"
#include "index/CanonicalIncludes.h"
#include "index/FileIndex.h"
#include "index/Index.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/Utils.h"
#include "clang/Index/IndexSymbol.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ::testing::_;
using ::testing::AllOf;
using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

MATCHER_P(RefRange, Range, "") {
  return std::make_tuple(arg.Location.Start.line(), arg.Location.Start.column(),
                         arg.Location.End.line(), arg.Location.End.column()) ==
         std::make_tuple(Range.start.line, Range.start.character,
                         Range.end.line, Range.end.character);
}
MATCHER_P(FileURI, F, "") { return llvm::StringRef(arg.Location.FileURI) == F; }
MATCHER_P(DeclURI, U, "") {
  return llvm::StringRef(arg.CanonicalDeclaration.FileURI) == U;
}
MATCHER_P(DefURI, U, "") {
  return llvm::StringRef(arg.Definition.FileURI) == U;
}
MATCHER_P(QName, N, "") { return (arg.Scope + arg.Name).str() == N; }

namespace clang {
namespace clangd {
namespace {
::testing::Matcher<const RefSlab &>
RefsAre(std::vector<::testing::Matcher<Ref>> Matchers) {
  return ElementsAre(::testing::Pair(_, UnorderedElementsAreArray(Matchers)));
}

Symbol symbol(llvm::StringRef ID) {
  Symbol Sym;
  Sym.ID = SymbolID(ID);
  Sym.Name = ID;
  return Sym;
}

std::unique_ptr<SymbolSlab> numSlab(int Begin, int End) {
  SymbolSlab::Builder Slab;
  for (int i = Begin; i <= End; i++)
    Slab.insert(symbol(std::to_string(i)));
  return llvm::make_unique<SymbolSlab>(std::move(Slab).build());
}

std::unique_ptr<RefSlab> refSlab(const SymbolID &ID, const char *Path) {
  RefSlab::Builder Slab;
  Ref R;
  R.Location.FileURI = Path;
  R.Kind = RefKind::Reference;
  Slab.insert(ID, R);
  return llvm::make_unique<RefSlab>(std::move(Slab).build());
}

TEST(FileSymbolsTest, UpdateAndGet) {
  FileSymbols FS;
  EXPECT_THAT(runFuzzyFind(*FS.buildIndex(IndexType::Light), ""), IsEmpty());

  FS.update("f1", numSlab(1, 3), refSlab(SymbolID("1"), "f1.cc"));
  EXPECT_THAT(runFuzzyFind(*FS.buildIndex(IndexType::Light), ""),
              UnorderedElementsAre(QName("1"), QName("2"), QName("3")));
  EXPECT_THAT(getRefs(*FS.buildIndex(IndexType::Light), SymbolID("1")),
              RefsAre({FileURI("f1.cc")}));
}

TEST(FileSymbolsTest, Overlap) {
  FileSymbols FS;
  FS.update("f1", numSlab(1, 3), nullptr);
  FS.update("f2", numSlab(3, 5), nullptr);
  for (auto Type : {IndexType::Light, IndexType::Heavy})
    EXPECT_THAT(runFuzzyFind(*FS.buildIndex(Type), ""),
                UnorderedElementsAre(QName("1"), QName("2"), QName("3"),
                                     QName("4"), QName("5")));
}

TEST(FileSymbolsTest, MergeOverlap) {
  FileSymbols FS;
  auto OneSymboSlab = [](Symbol Sym) {
    SymbolSlab::Builder S;
    S.insert(Sym);
    return llvm::make_unique<SymbolSlab>(std::move(S).build());
  };
  auto X1 = symbol("x");
  X1.CanonicalDeclaration.FileURI = "file:///x1";
  auto X2 = symbol("x");
  X2.Definition.FileURI = "file:///x2";

  FS.update("f1", OneSymboSlab(X1), nullptr);
  FS.update("f2", OneSymboSlab(X2), nullptr);
  for (auto Type : {IndexType::Light, IndexType::Heavy})
    EXPECT_THAT(
        runFuzzyFind(*FS.buildIndex(Type, DuplicateHandling::Merge), "x"),
        UnorderedElementsAre(
            AllOf(QName("x"), DeclURI("file:///x1"), DefURI("file:///x2"))));
}

TEST(FileSymbolsTest, SnapshotAliveAfterRemove) {
  FileSymbols FS;

  SymbolID ID("1");
  FS.update("f1", numSlab(1, 3), refSlab(ID, "f1.cc"));

  auto Symbols = FS.buildIndex(IndexType::Light);
  EXPECT_THAT(runFuzzyFind(*Symbols, ""),
              UnorderedElementsAre(QName("1"), QName("2"), QName("3")));
  EXPECT_THAT(getRefs(*Symbols, ID), RefsAre({FileURI("f1.cc")}));

  FS.update("f1", nullptr, nullptr);
  auto Empty = FS.buildIndex(IndexType::Light);
  EXPECT_THAT(runFuzzyFind(*Empty, ""), IsEmpty());
  EXPECT_THAT(getRefs(*Empty, ID), ElementsAre());

  EXPECT_THAT(runFuzzyFind(*Symbols, ""),
              UnorderedElementsAre(QName("1"), QName("2"), QName("3")));
  EXPECT_THAT(getRefs(*Symbols, ID), RefsAre({FileURI("f1.cc")}));
}

// Adds Basename.cpp, which includes Basename.h, which contains Code.
void update(FileIndex &M, llvm::StringRef Basename, llvm::StringRef Code) {
  TestTU File;
  File.Filename = (Basename + ".cpp").str();
  File.HeaderFilename = (Basename + ".h").str();
  File.HeaderCode = Code;
  auto AST = File.build();
  M.updatePreamble(File.Filename, AST.getASTContext(), AST.getPreprocessorPtr(),
                   AST.getCanonicalIncludes());
}

TEST(FileIndexTest, CustomizedURIScheme) {
  FileIndex M;
  update(M, "f", "class string {};");

  EXPECT_THAT(runFuzzyFind(M, ""), ElementsAre(DeclURI("unittest:///f.h")));
}

TEST(FileIndexTest, IndexAST) {
  FileIndex M;
  update(M, "f1", "namespace ns { void f() {} class X {}; }");

  FuzzyFindRequest Req;
  Req.Query = "";
  Req.Scopes = {"ns::"};
  EXPECT_THAT(runFuzzyFind(M, Req),
              UnorderedElementsAre(QName("ns::f"), QName("ns::X")));
}

TEST(FileIndexTest, NoLocal) {
  FileIndex M;
  update(M, "f1", "namespace ns { void f() { int local = 0; } class X {}; }");

  EXPECT_THAT(
      runFuzzyFind(M, ""),
      UnorderedElementsAre(QName("ns"), QName("ns::f"), QName("ns::X")));
}

TEST(FileIndexTest, IndexMultiASTAndDeduplicate) {
  FileIndex M;
  update(M, "f1", "namespace ns { void f() {} class X {}; }");
  update(M, "f2", "namespace ns { void ff() {} class X {}; }");

  FuzzyFindRequest Req;
  Req.Scopes = {"ns::"};
  EXPECT_THAT(
      runFuzzyFind(M, Req),
      UnorderedElementsAre(QName("ns::f"), QName("ns::X"), QName("ns::ff")));
}

TEST(FileIndexTest, ClassMembers) {
  FileIndex M;
  update(M, "f1", "class X { static int m1; int m2; static void f(); };");

  EXPECT_THAT(runFuzzyFind(M, ""),
              UnorderedElementsAre(QName("X"), QName("X::m1"), QName("X::m2"),
                                   QName("X::f")));
}

TEST(FileIndexTest, IncludeCollected) {
  FileIndex M;
  update(
      M, "f",
      "// IWYU pragma: private, include <the/good/header.h>\nclass string {};");

  auto Symbols = runFuzzyFind(M, "");
  EXPECT_THAT(Symbols, ElementsAre(_));
  EXPECT_THAT(Symbols.begin()->IncludeHeaders.front().IncludeHeader,
              "<the/good/header.h>");
}

TEST(FileIndexTest, HasSystemHeaderMappingsInPreamble) {
  TestTU TU;
  TU.HeaderCode = "class Foo{};";
  TU.HeaderFilename = "algorithm";

  auto Symbols = runFuzzyFind(*TU.index(), "");
  EXPECT_THAT(Symbols, ElementsAre(_));
  EXPECT_THAT(Symbols.begin()->IncludeHeaders.front().IncludeHeader,
              "<algorithm>");
}

TEST(FileIndexTest, TemplateParamsInLabel) {
  auto Source = R"cpp(
template <class Ty>
class vector {
};

template <class Ty, class Arg>
vector<Ty> make_vector(Arg A) {}
)cpp";

  FileIndex M;
  update(M, "f", Source);

  auto Symbols = runFuzzyFind(M, "");
  EXPECT_THAT(Symbols,
              UnorderedElementsAre(QName("vector"), QName("make_vector")));
  auto It = Symbols.begin();
  Symbol Vector = *It++;
  Symbol MakeVector = *It++;
  if (MakeVector.Name == "vector")
    std::swap(MakeVector, Vector);

  EXPECT_EQ(Vector.Signature, "<class Ty>");
  EXPECT_EQ(Vector.CompletionSnippetSuffix, "<${1:class Ty}>");

  EXPECT_EQ(MakeVector.Signature, "<class Ty>(Arg A)");
  EXPECT_EQ(MakeVector.CompletionSnippetSuffix, "<${1:class Ty}>(${2:Arg A})");
}

TEST(FileIndexTest, RebuildWithPreamble) {
  auto FooCpp = testPath("foo.cpp");
  auto FooH = testPath("foo.h");
  // Preparse ParseInputs.
  ParseInputs PI;
  PI.CompileCommand.Directory = testRoot();
  PI.CompileCommand.Filename = FooCpp;
  PI.CompileCommand.CommandLine = {"clang", "-xc++", FooCpp};

  llvm::StringMap<std::string> Files;
  Files[FooCpp] = "";
  Files[FooH] = R"cpp(
    namespace ns_in_header {
      int func_in_header();
    }
  )cpp";
  PI.FS = buildTestFS(std::move(Files));

  PI.Contents = R"cpp(
    #include "foo.h"
    namespace ns_in_source {
      int func_in_source();
    }
  )cpp";

  // Rebuild the file.
  auto CI = buildCompilerInvocation(PI);

  FileIndex Index;
  bool IndexUpdated = false;
  buildPreamble(
      FooCpp, *CI, /*OldPreamble=*/nullptr, tooling::CompileCommand(), PI,
      /*StoreInMemory=*/true,
      [&](ASTContext &Ctx, std::shared_ptr<Preprocessor> PP,
          const CanonicalIncludes &CanonIncludes) {
        EXPECT_FALSE(IndexUpdated) << "Expected only a single index update";
        IndexUpdated = true;
        Index.updatePreamble(FooCpp, Ctx, std::move(PP), CanonIncludes);
      });
  ASSERT_TRUE(IndexUpdated);

  // Check the index contains symbols from the preamble, but not from the main
  // file.
  FuzzyFindRequest Req;
  Req.Query = "";
  Req.Scopes = {"", "ns_in_header::"};

  EXPECT_THAT(runFuzzyFind(Index, Req),
              UnorderedElementsAre(QName("ns_in_header"),
                                   QName("ns_in_header::func_in_header")));
}

TEST(FileIndexTest, Refs) {
  const char *HeaderCode = "class Foo {};";
  Annotations MainCode(R"cpp(
  void f() {
    $foo[[Foo]] foo;
  }
  )cpp");

  auto Foo =
      findSymbol(TestTU::withHeaderCode(HeaderCode).headerSymbols(), "Foo");

  RefsRequest Request;
  Request.IDs = {Foo.ID};

  FileIndex Index;
  // Add test.cc
  TestTU Test;
  Test.HeaderCode = HeaderCode;
  Test.Code = MainCode.code();
  Test.Filename = "test.cc";
  auto AST = Test.build();
  Index.updateMain(Test.Filename, AST);
  // Add test2.cc
  TestTU Test2;
  Test2.HeaderCode = HeaderCode;
  Test2.Code = MainCode.code();
  Test2.Filename = "test2.cc";
  AST = Test2.build();
  Index.updateMain(Test2.Filename, AST);

  EXPECT_THAT(getRefs(Index, Foo.ID),
              RefsAre({AllOf(RefRange(MainCode.range("foo")),
                             FileURI("unittest:///test.cc")),
                       AllOf(RefRange(MainCode.range("foo")),
                             FileURI("unittest:///test2.cc"))}));
}

TEST(FileIndexTest, CollectMacros) {
  FileIndex M;
  update(M, "f", "#define CLANGD 1");
  EXPECT_THAT(runFuzzyFind(M, ""), Contains(QName("CLANGD")));
}

TEST(FileIndexTest, ReferencesInMainFileWithPreamble) {
  TestTU TU;
  TU.HeaderCode = "class Foo{};";
  Annotations Main(R"cpp(
    #include "foo.h"
    void f() {
      [[Foo]] foo;
    }
  )cpp");
  TU.Code = Main.code();
  auto AST = TU.build();
  FileIndex Index;
  Index.updateMain(testPath(TU.Filename), AST);

  // Expect to see references in main file, references in headers are excluded
  // because we only index main AST.
  EXPECT_THAT(getRefs(Index, findSymbol(TU.headerSymbols(), "Foo").ID),
              RefsAre({RefRange(Main.range())}));
}

} // namespace
} // namespace clangd
} // namespace clang
