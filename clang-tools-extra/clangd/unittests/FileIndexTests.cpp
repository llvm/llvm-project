//===-- FileIndexTests.cpp  ---------------------------*- C++ -*-----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "Compiler.h"
#include "Headers.h"
#include "ParsedAST.h"
#include "SyncAPI.h"
#include "TestFS.h"
#include "TestTU.h"
#include "TestWorkspace.h"
#include "URI.h"
#include "clang-include-cleaner/Record.h"
#include "index/FileIndex.h"
#include "index/Index.h"
#include "index/Ref.h"
#include "index/Relation.h"
#include "index/Serialization.h"
#include "index/Symbol.h"
#include "index/SymbolID.h"
#include "support/Threading.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Allocator.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>
#include <utility>
#include <vector>

using ::testing::_;
using ::testing::AllOf;
using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::Gt;
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

MATCHER_P(refRange, Range, "") {
  return std::make_tuple(arg.Location.Start.line(), arg.Location.Start.column(),
                         arg.Location.End.line(), arg.Location.End.column()) ==
         std::make_tuple(Range.start.line, Range.start.character,
                         Range.end.line, Range.end.character);
}
MATCHER_P(fileURI, F, "") { return llvm::StringRef(arg.Location.FileURI) == F; }
MATCHER_P(declURI, U, "") {
  return llvm::StringRef(arg.CanonicalDeclaration.FileURI) == U;
}
MATCHER_P(defURI, U, "") {
  return llvm::StringRef(arg.Definition.FileURI) == U;
}
MATCHER_P(qName, N, "") { return (arg.Scope + arg.Name).str() == N; }
MATCHER_P(numReferences, N, "") { return arg.References == N; }
MATCHER_P(hasOrign, O, "") { return bool(arg.Origin & O); }

MATCHER_P(includeHeader, P, "") {
  return (arg.IncludeHeaders.size() == 1) &&
         (arg.IncludeHeaders.begin()->IncludeHeader == P);
}

namespace clang {
namespace clangd {
namespace {
::testing::Matcher<const RefSlab &>
refsAre(std::vector<::testing::Matcher<Ref>> Matchers) {
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
  for (int I = Begin; I <= End; I++)
    Slab.insert(symbol(std::to_string(I)));
  return std::make_unique<SymbolSlab>(std::move(Slab).build());
}

std::unique_ptr<RefSlab> refSlab(const SymbolID &ID, const char *Path) {
  RefSlab::Builder Slab;
  Ref R;
  R.Location.FileURI = Path;
  R.Kind = RefKind::Reference;
  Slab.insert(ID, R);
  return std::make_unique<RefSlab>(std::move(Slab).build());
}

std::unique_ptr<RelationSlab> relSlab(llvm::ArrayRef<const Relation> Rels) {
  RelationSlab::Builder RelBuilder;
  for (auto &Rel : Rels)
    RelBuilder.insert(Rel);
  return std::make_unique<RelationSlab>(std::move(RelBuilder).build());
}

TEST(FileSymbolsTest, UpdateAndGet) {
  FileSymbols FS(IndexContents::All);
  EXPECT_THAT(runFuzzyFind(*FS.buildIndex(IndexType::Light), ""), IsEmpty());

  FS.update("f1", numSlab(1, 3), refSlab(SymbolID("1"), "f1.cc"), nullptr,
            false);
  EXPECT_THAT(runFuzzyFind(*FS.buildIndex(IndexType::Light), ""),
              UnorderedElementsAre(qName("1"), qName("2"), qName("3")));
  EXPECT_THAT(getRefs(*FS.buildIndex(IndexType::Light), SymbolID("1")),
              refsAre({fileURI("f1.cc")}));
}

TEST(FileSymbolsTest, Overlap) {
  FileSymbols FS(IndexContents::All);
  FS.update("f1", numSlab(1, 3), nullptr, nullptr, false);
  FS.update("f2", numSlab(3, 5), nullptr, nullptr, false);
  for (auto Type : {IndexType::Light, IndexType::Heavy})
    EXPECT_THAT(runFuzzyFind(*FS.buildIndex(Type), ""),
                UnorderedElementsAre(qName("1"), qName("2"), qName("3"),
                                     qName("4"), qName("5")));
}

TEST(FileSymbolsTest, MergeOverlap) {
  FileSymbols FS(IndexContents::All);
  auto OneSymboSlab = [](Symbol Sym) {
    SymbolSlab::Builder S;
    S.insert(Sym);
    return std::make_unique<SymbolSlab>(std::move(S).build());
  };
  auto X1 = symbol("x");
  X1.CanonicalDeclaration.FileURI = "file:///x1";
  auto X2 = symbol("x");
  X2.Definition.FileURI = "file:///x2";

  FS.update("f1", OneSymboSlab(X1), nullptr, nullptr, false);
  FS.update("f2", OneSymboSlab(X2), nullptr, nullptr, false);
  for (auto Type : {IndexType::Light, IndexType::Heavy})
    EXPECT_THAT(
        runFuzzyFind(*FS.buildIndex(Type, DuplicateHandling::Merge), "x"),
        UnorderedElementsAre(
            AllOf(qName("x"), declURI("file:///x1"), defURI("file:///x2"))));
}

TEST(FileSymbolsTest, SnapshotAliveAfterRemove) {
  FileSymbols FS(IndexContents::All);

  SymbolID ID("1");
  FS.update("f1", numSlab(1, 3), refSlab(ID, "f1.cc"), nullptr, false);

  auto Symbols = FS.buildIndex(IndexType::Light);
  EXPECT_THAT(runFuzzyFind(*Symbols, ""),
              UnorderedElementsAre(qName("1"), qName("2"), qName("3")));
  EXPECT_THAT(getRefs(*Symbols, ID), refsAre({fileURI("f1.cc")}));

  FS.update("f1", nullptr, nullptr, nullptr, false);
  auto Empty = FS.buildIndex(IndexType::Light);
  EXPECT_THAT(runFuzzyFind(*Empty, ""), IsEmpty());
  EXPECT_THAT(getRefs(*Empty, ID), ElementsAre());

  EXPECT_THAT(runFuzzyFind(*Symbols, ""),
              UnorderedElementsAre(qName("1"), qName("2"), qName("3")));
  EXPECT_THAT(getRefs(*Symbols, ID), refsAre({fileURI("f1.cc")}));
}

// Adds Basename.cpp, which includes Basename.h, which contains Code.
void update(FileIndex &M, llvm::StringRef Basename, llvm::StringRef Code) {
  TestTU File;
  File.Filename = (Basename + ".cpp").str();
  File.HeaderFilename = (Basename + ".h").str();
  File.HeaderCode = std::string(Code);
  auto AST = File.build();
  M.updatePreamble(testPath(File.Filename), /*Version=*/"null",
                   AST.getASTContext(), AST.getPreprocessor(),
                   *AST.getPragmaIncludes());
}

TEST(FileIndexTest, CustomizedURIScheme) {
  FileIndex M;
  update(M, "f", "class string {};");

  EXPECT_THAT(runFuzzyFind(M, ""), ElementsAre(declURI("unittest:///f.h")));
}

TEST(FileIndexTest, IndexAST) {
  FileIndex M;
  update(M, "f1", "namespace ns { void f() {} class X {}; }");

  FuzzyFindRequest Req;
  Req.Query = "";
  Req.Scopes = {"ns::"};
  EXPECT_THAT(runFuzzyFind(M, Req),
              UnorderedElementsAre(qName("ns::f"), qName("ns::X")));
}

TEST(FileIndexTest, NoLocal) {
  FileIndex M;
  update(M, "f1", "namespace ns { void f() { int local = 0; } class X {}; }");

  EXPECT_THAT(
      runFuzzyFind(M, ""),
      UnorderedElementsAre(qName("ns"), qName("ns::f"), qName("ns::X")));
}

TEST(FileIndexTest, IndexMultiASTAndDeduplicate) {
  FileIndex M;
  update(M, "f1", "namespace ns { void f() {} class X {}; }");
  update(M, "f2", "namespace ns { void ff() {} class X {}; }");

  FuzzyFindRequest Req;
  Req.Scopes = {"ns::"};
  EXPECT_THAT(
      runFuzzyFind(M, Req),
      UnorderedElementsAre(qName("ns::f"), qName("ns::X"), qName("ns::ff")));
}

TEST(FileIndexTest, ClassMembers) {
  FileIndex M;
  update(M, "f1", "class X { static int m1; int m2; static void f(); };");

  EXPECT_THAT(runFuzzyFind(M, ""),
              UnorderedElementsAre(qName("X"), qName("X::m1"), qName("X::m2"),
                                   qName("X::f")));
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

TEST(FileIndexTest, DISABLED_IWYUPragmaExport) {
  FileIndex M;

  TestTU File;
  File.Code = R"cpp(#pragma once
    #include "exporter.h"
  )cpp";
  File.HeaderFilename = "exporter.h";
  File.HeaderCode = R"cpp(#pragma once
    #include "private.h" // IWYU pragma: export
  )cpp";
  File.AdditionalFiles["private.h"] = "class Foo{};";
  auto AST = File.build();
  M.updatePreamble(testPath(File.Filename), /*Version=*/"null",
                   AST.getASTContext(), AST.getPreprocessor(),
                   *AST.getPragmaIncludes());

  auto Symbols = runFuzzyFind(M, "");
  EXPECT_THAT(
      Symbols,
      UnorderedElementsAre(AllOf(
          qName("Foo"),
          includeHeader(URI::create(testPath(File.HeaderFilename)).toString()),
          declURI(URI::create(testPath("private.h")).toString()))));
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
  auto *Source = R"cpp(
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
              UnorderedElementsAre(qName("vector"), qName("make_vector")));
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

  MockFS FS;
  FS.Files[FooCpp] = "";
  FS.Files[FooH] = R"cpp(
    namespace ns_in_header {
      int func_in_header();
    }
  )cpp";
  PI.TFS = &FS;

  PI.Contents = R"cpp(
    #include "foo.h"
    namespace ns_in_source {
      int func_in_source();
    }
  )cpp";

  // Rebuild the file.
  IgnoreDiagnostics IgnoreDiags;
  auto CI = buildCompilerInvocation(PI, IgnoreDiags);

  FileIndex Index;
  bool IndexUpdated = false;
  buildPreamble(
      FooCpp, *CI, PI,
      /*StoreInMemory=*/true,
      [&](CapturedASTCtx ASTCtx,
          std::shared_ptr<const include_cleaner::PragmaIncludes> PI) {
        auto &Ctx = ASTCtx.getASTContext();
        auto &PP = ASTCtx.getPreprocessor();
        EXPECT_FALSE(IndexUpdated) << "Expected only a single index update";
        IndexUpdated = true;
        Index.updatePreamble(FooCpp, /*Version=*/"null", Ctx, PP, *PI);
      });
  ASSERT_TRUE(IndexUpdated);

  // Check the index contains symbols from the preamble, but not from the main
  // file.
  FuzzyFindRequest Req;
  Req.Query = "";
  Req.Scopes = {"", "ns_in_header::"};

  EXPECT_THAT(runFuzzyFind(Index, Req),
              UnorderedElementsAre(qName("ns_in_header"),
                                   qName("ns_in_header::func_in_header")));
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
  Test.Code = std::string(MainCode.code());
  Test.Filename = "test.cc";
  auto AST = Test.build();
  Index.updateMain(testPath(Test.Filename), AST);
  // Add test2.cc
  TestTU Test2;
  Test2.HeaderCode = HeaderCode;
  Test2.Code = std::string(MainCode.code());
  Test2.Filename = "test2.cc";
  AST = Test2.build();
  Index.updateMain(testPath(Test2.Filename), AST);

  EXPECT_THAT(getRefs(Index, Foo.ID),
              refsAre({AllOf(refRange(MainCode.range("foo")),
                             fileURI("unittest:///test.cc")),
                       AllOf(refRange(MainCode.range("foo")),
                             fileURI("unittest:///test2.cc"))}));
}

TEST(FileIndexTest, MacroRefs) {
  Annotations HeaderCode(R"cpp(
    #define $def1[[HEADER_MACRO]](X) (X+1)
  )cpp");
  Annotations MainCode(R"cpp(
  #define $def2[[MAINFILE_MACRO]](X) (X+1)
  void f() {
    int a = $ref1[[HEADER_MACRO]](2);
    int b = $ref2[[MAINFILE_MACRO]](1);
  }
  )cpp");

  FileIndex Index;
  // Add test.cc
  TestTU Test;
  Test.HeaderCode = std::string(HeaderCode.code());
  Test.Code = std::string(MainCode.code());
  Test.Filename = "test.cc";
  auto AST = Test.build();
  Index.updateMain(testPath(Test.Filename), AST);

  auto HeaderMacro = findSymbol(Test.headerSymbols(), "HEADER_MACRO");
  EXPECT_THAT(getRefs(Index, HeaderMacro.ID),
              refsAre({AllOf(refRange(MainCode.range("ref1")),
                             fileURI("unittest:///test.cc"))}));

  auto MainFileMacro = findSymbol(Test.headerSymbols(), "MAINFILE_MACRO");
  EXPECT_THAT(getRefs(Index, MainFileMacro.ID),
              refsAre({AllOf(refRange(MainCode.range("def2")),
                             fileURI("unittest:///test.cc")),
                       AllOf(refRange(MainCode.range("ref2")),
                             fileURI("unittest:///test.cc"))}));
}

TEST(FileIndexTest, CollectMacros) {
  FileIndex M;
  update(M, "f", "#define CLANGD 1");
  EXPECT_THAT(runFuzzyFind(M, ""), Contains(qName("CLANGD")));
}

TEST(FileIndexTest, Relations) {
  TestTU TU;
  TU.Filename = "f.cpp";
  TU.HeaderFilename = "f.h";
  TU.HeaderCode = "class A {}; class B : public A {};";
  auto AST = TU.build();
  FileIndex Index;
  Index.updatePreamble(testPath(TU.Filename), /*Version=*/"null",
                       AST.getASTContext(), AST.getPreprocessor(),
                       *AST.getPragmaIncludes());
  SymbolID A = findSymbol(TU.headerSymbols(), "A").ID;
  uint32_t Results = 0;
  RelationsRequest Req;
  Req.Subjects.insert(A);
  Req.Predicate = RelationKind::BaseOf;
  Index.relations(Req, [&](const SymbolID &, const Symbol &) { ++Results; });
  EXPECT_EQ(Results, 1u);
}

TEST(FileIndexTest, RelationsMultiFile) {
  TestWorkspace Workspace;
  Workspace.addSource("Base.h", "class Base {};");
  Workspace.addMainFile("A.cpp", R"cpp(
    #include "Base.h"
    class A : public Base {};
  )cpp");
  Workspace.addMainFile("B.cpp", R"cpp(
    #include "Base.h"
    class B : public Base {};
  )cpp");

  auto Index = Workspace.index();
  FuzzyFindRequest FFReq;
  FFReq.Query = "Base";
  FFReq.AnyScope = true;
  SymbolID Base;
  Index->fuzzyFind(FFReq, [&](const Symbol &S) { Base = S.ID; });

  RelationsRequest Req;
  Req.Subjects.insert(Base);
  Req.Predicate = RelationKind::BaseOf;
  uint32_t Results = 0;
  Index->relations(Req, [&](const SymbolID &, const Symbol &) { ++Results; });
  EXPECT_EQ(Results, 2u);
}

TEST(FileIndexTest, ReferencesInMainFileWithPreamble) {
  TestTU TU;
  TU.HeaderCode = "class Foo{};";
  Annotations Main(R"cpp(
    void f() {
      [[Foo]] foo;
    }
  )cpp");
  TU.Code = std::string(Main.code());
  auto AST = TU.build();
  FileIndex Index;
  Index.updateMain(testPath(TU.Filename), AST);

  // Expect to see references in main file, references in headers are excluded
  // because we only index main AST.
  EXPECT_THAT(getRefs(Index, findSymbol(TU.headerSymbols(), "Foo").ID),
              refsAre({refRange(Main.range())}));
}

TEST(FileIndexTest, MergeMainFileSymbols) {
  const char *CommonHeader = "void foo();";
  TestTU Header = TestTU::withCode(CommonHeader);
  TestTU Cpp = TestTU::withCode("void foo() {}");
  Cpp.Filename = "foo.cpp";
  Cpp.HeaderFilename = "foo.h";
  Cpp.HeaderCode = CommonHeader;

  FileIndex Index;
  auto HeaderAST = Header.build();
  auto CppAST = Cpp.build();
  Index.updateMain(testPath("foo.h"), HeaderAST);
  Index.updateMain(testPath("foo.cpp"), CppAST);

  auto Symbols = runFuzzyFind(Index, "");
  // Check foo is merged, foo in Cpp wins (as we see the definition there).
  EXPECT_THAT(Symbols, ElementsAre(AllOf(declURI("unittest:///foo.h"),
                                         defURI("unittest:///foo.cpp"),
                                         hasOrign(SymbolOrigin::Merge))));
}

TEST(FileSymbolsTest, CountReferencesNoRefSlabs) {
  FileSymbols FS(IndexContents::All);
  FS.update("f1", numSlab(1, 3), nullptr, nullptr, true);
  FS.update("f2", numSlab(1, 3), nullptr, nullptr, false);
  EXPECT_THAT(
      runFuzzyFind(*FS.buildIndex(IndexType::Light, DuplicateHandling::Merge),
                   ""),
      UnorderedElementsAre(AllOf(qName("1"), numReferences(0u)),
                           AllOf(qName("2"), numReferences(0u)),
                           AllOf(qName("3"), numReferences(0u))));
}

TEST(FileSymbolsTest, CountReferencesWithRefSlabs) {
  FileSymbols FS(IndexContents::All);
  FS.update("f1cpp", numSlab(1, 3), refSlab(SymbolID("1"), "f1.cpp"), nullptr,
            true);
  FS.update("f1h", numSlab(1, 3), refSlab(SymbolID("1"), "f1.h"), nullptr,
            false);
  FS.update("f2cpp", numSlab(1, 3), refSlab(SymbolID("2"), "f2.cpp"), nullptr,
            true);
  FS.update("f2h", numSlab(1, 3), refSlab(SymbolID("2"), "f2.h"), nullptr,
            false);
  FS.update("f3cpp", numSlab(1, 3), refSlab(SymbolID("3"), "f3.cpp"), nullptr,
            true);
  FS.update("f3h", numSlab(1, 3), refSlab(SymbolID("3"), "f3.h"), nullptr,
            false);
  EXPECT_THAT(
      runFuzzyFind(*FS.buildIndex(IndexType::Light, DuplicateHandling::Merge),
                   ""),
      UnorderedElementsAre(AllOf(qName("1"), numReferences(1u)),
                           AllOf(qName("2"), numReferences(1u)),
                           AllOf(qName("3"), numReferences(1u))));
}

TEST(FileIndexTest, StalePreambleSymbolsDeleted) {
  FileIndex M;
  TestTU File;
  File.HeaderFilename = "a.h";

  File.Filename = "f1.cpp";
  File.HeaderCode = "int a;";
  auto AST = File.build();
  M.updatePreamble(testPath(File.Filename), /*Version=*/"null",
                   AST.getASTContext(), AST.getPreprocessor(),
                   *AST.getPragmaIncludes());
  EXPECT_THAT(runFuzzyFind(M, ""), UnorderedElementsAre(qName("a")));

  File.Filename = "f2.cpp";
  File.HeaderCode = "int b;";
  AST = File.build();
  M.updatePreamble(testPath(File.Filename), /*Version=*/"null",
                   AST.getASTContext(), AST.getPreprocessor(),
                   *AST.getPragmaIncludes());
  EXPECT_THAT(runFuzzyFind(M, ""), UnorderedElementsAre(qName("b")));
}

// Verifies that concurrent calls to updateMain don't "lose" any updates.
TEST(FileIndexTest, Threadsafety) {
  FileIndex M;
  Notification Go;

  constexpr int Count = 10;
  {
    // Set up workers to concurrently call updateMain() with separate files.
    AsyncTaskRunner Pool;
    for (unsigned I = 0; I < Count; ++I) {
      auto TU = TestTU::withCode(llvm::formatv("int xxx{0};", I).str());
      TU.Filename = llvm::formatv("x{0}.c", I).str();
      Pool.runAsync(TU.Filename, [&, Filename(testPath(TU.Filename)),
                                  AST(TU.build())]() mutable {
        Go.wait();
        M.updateMain(Filename, AST);
      });
    }
    // On your marks, get set...
    Go.notify();
  }

  EXPECT_THAT(runFuzzyFind(M, "xxx"), ::testing::SizeIs(Count));
}

TEST(FileShardedIndexTest, Sharding) {
  auto AHeaderUri = URI::create(testPath("a.h")).toString();
  auto BHeaderUri = URI::create(testPath("b.h")).toString();
  auto BSourceUri = URI::create(testPath("b.cc")).toString();

  auto Sym1 = symbol("1");
  Sym1.CanonicalDeclaration.FileURI = AHeaderUri.c_str();

  auto Sym2 = symbol("2");
  Sym2.CanonicalDeclaration.FileURI = BHeaderUri.c_str();
  Sym2.Definition.FileURI = BSourceUri.c_str();

  auto Sym3 = symbol("3"); // not stored

  IndexFileIn IF;
  {
    SymbolSlab::Builder B;
    // Should be stored in only a.h
    B.insert(Sym1);
    // Should be stored in both b.h and b.cc
    B.insert(Sym2);
    IF.Symbols.emplace(std::move(B).build());
  }
  {
    // Should be stored in b.cc
    IF.Refs.emplace(std::move(*refSlab(Sym1.ID, BSourceUri.c_str())));
  }
  {
    RelationSlab::Builder B;
    // Should be stored in a.h and b.h
    B.insert(Relation{Sym1.ID, RelationKind::BaseOf, Sym2.ID});
    // Should be stored in a.h and b.h
    B.insert(Relation{Sym2.ID, RelationKind::BaseOf, Sym1.ID});
    // Should be stored in a.h (where Sym1 is stored) even though
    // the relation is dangling as Sym3 is unknown.
    B.insert(Relation{Sym3.ID, RelationKind::BaseOf, Sym1.ID});
    IF.Relations.emplace(std::move(B).build());
  }

  IF.Sources.emplace();
  IncludeGraph &IG = *IF.Sources;
  {
    // b.cc includes b.h
    auto &Node = IG[BSourceUri];
    Node.DirectIncludes = {BHeaderUri};
    Node.URI = BSourceUri;
  }
  {
    // b.h includes a.h
    auto &Node = IG[BHeaderUri];
    Node.DirectIncludes = {AHeaderUri};
    Node.URI = BHeaderUri;
  }
  {
    // a.h includes nothing.
    auto &Node = IG[AHeaderUri];
    Node.DirectIncludes = {};
    Node.URI = AHeaderUri;
  }

  IF.Cmd = tooling::CompileCommand(testRoot(), "b.cc", {"clang"}, "out");

  FileShardedIndex ShardedIndex(std::move(IF));
  ASSERT_THAT(ShardedIndex.getAllSources(),
              UnorderedElementsAre(AHeaderUri, BHeaderUri, BSourceUri));

  {
    auto Shard = ShardedIndex.getShard(AHeaderUri);
    ASSERT_TRUE(Shard);
    EXPECT_THAT(*Shard->Symbols, UnorderedElementsAre(qName("1")));
    EXPECT_THAT(*Shard->Refs, IsEmpty());
    EXPECT_THAT(
        *Shard->Relations,
        UnorderedElementsAre(Relation{Sym1.ID, RelationKind::BaseOf, Sym2.ID},
                             Relation{Sym2.ID, RelationKind::BaseOf, Sym1.ID},
                             Relation{Sym3.ID, RelationKind::BaseOf, Sym1.ID}));
    ASSERT_THAT(Shard->Sources->keys(), UnorderedElementsAre(AHeaderUri));
    EXPECT_THAT(Shard->Sources->lookup(AHeaderUri).DirectIncludes, IsEmpty());
    EXPECT_TRUE(Shard->Cmd);
  }
  {
    auto Shard = ShardedIndex.getShard(BHeaderUri);
    ASSERT_TRUE(Shard);
    EXPECT_THAT(*Shard->Symbols, UnorderedElementsAre(qName("2")));
    EXPECT_THAT(*Shard->Refs, IsEmpty());
    EXPECT_THAT(
        *Shard->Relations,
        UnorderedElementsAre(Relation{Sym1.ID, RelationKind::BaseOf, Sym2.ID},
                             Relation{Sym2.ID, RelationKind::BaseOf, Sym1.ID}));
    ASSERT_THAT(Shard->Sources->keys(),
                UnorderedElementsAre(BHeaderUri, AHeaderUri));
    EXPECT_THAT(Shard->Sources->lookup(BHeaderUri).DirectIncludes,
                UnorderedElementsAre(AHeaderUri));
    EXPECT_TRUE(Shard->Cmd);
  }
  {
    auto Shard = ShardedIndex.getShard(BSourceUri);
    ASSERT_TRUE(Shard);
    EXPECT_THAT(*Shard->Symbols, UnorderedElementsAre(qName("2")));
    EXPECT_THAT(*Shard->Refs, UnorderedElementsAre(Pair(Sym1.ID, _)));
    EXPECT_THAT(*Shard->Relations, IsEmpty());
    ASSERT_THAT(Shard->Sources->keys(),
                UnorderedElementsAre(BSourceUri, BHeaderUri));
    EXPECT_THAT(Shard->Sources->lookup(BSourceUri).DirectIncludes,
                UnorderedElementsAre(BHeaderUri));
    EXPECT_TRUE(Shard->Cmd);
  }
}

TEST(FileIndexTest, Profile) {
  FileIndex FI;

  auto FileName = testPath("foo.cpp");
  auto AST = TestTU::withHeaderCode("int a;").build();
  FI.updateMain(FileName, AST);
  FI.updatePreamble(FileName, "v1", AST.getASTContext(), AST.getPreprocessor(),
                    *AST.getPragmaIncludes());

  llvm::BumpPtrAllocator Alloc;
  MemoryTree MT(&Alloc);
  FI.profile(MT);
  ASSERT_THAT(MT.children(),
              UnorderedElementsAre(Pair("preamble", _), Pair("main_file", _)));

  ASSERT_THAT(MT.child("preamble").children(),
              UnorderedElementsAre(Pair("index", _), Pair("slabs", _)));
  ASSERT_THAT(MT.child("main_file").children(),
              UnorderedElementsAre(Pair("index", _), Pair("slabs", _)));

  ASSERT_THAT(MT.child("preamble").child("index").total(), Gt(0U));
  ASSERT_THAT(MT.child("main_file").child("index").total(), Gt(0U));
}

TEST(FileSymbolsTest, Profile) {
  FileSymbols FS(IndexContents::All);
  FS.update("f1", numSlab(1, 2), nullptr, nullptr, false);
  FS.update("f2", nullptr, refSlab(SymbolID("1"), "f1"), nullptr, false);
  FS.update("f3", nullptr, nullptr,
            relSlab({{SymbolID("1"), RelationKind::BaseOf, SymbolID("2")}}),
            false);
  llvm::BumpPtrAllocator Alloc;
  MemoryTree MT(&Alloc);
  FS.profile(MT);
  ASSERT_THAT(MT.children(), UnorderedElementsAre(Pair("f1", _), Pair("f2", _),
                                                  Pair("f3", _)));
  EXPECT_THAT(MT.child("f1").children(), ElementsAre(Pair("symbols", _)));
  EXPECT_THAT(MT.child("f1").total(), Gt(0U));
  EXPECT_THAT(MT.child("f2").children(), ElementsAre(Pair("references", _)));
  EXPECT_THAT(MT.child("f2").total(), Gt(0U));
  EXPECT_THAT(MT.child("f3").children(), ElementsAre(Pair("relations", _)));
  EXPECT_THAT(MT.child("f3").total(), Gt(0U));
}

TEST(FileIndexTest, MacrosFromMainFile) {
  FileIndex Idx;
  TestTU TU;
  TU.Code = "#pragma once\n#define FOO";
  TU.Filename = "foo.h";
  auto AST = TU.build();
  Idx.updateMain(testPath(TU.Filename), AST);

  auto Slab = runFuzzyFind(Idx, "");
  auto &FooSymbol = findSymbol(Slab, "FOO");
  EXPECT_TRUE(FooSymbol.Flags & Symbol::IndexedForCodeCompletion);
}

} // namespace
} // namespace clangd
} // namespace clang
