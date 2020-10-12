//===-- FileIndexTests.cpp  ---------------------------*- C++ -*-----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AST.h"
#include "Annotations.h"
#include "Compiler.h"
#include "Headers.h"
#include "ParsedAST.h"
#include "SyncAPI.h"
#include "TestFS.h"
#include "TestTU.h"
#include "URI.h"
#include "index/CanonicalIncludes.h"
#include "index/FileIndex.h"
#include "index/Index.h"
#include "index/Ref.h"
#include "index/Relation.h"
#include "index/Serialization.h"
#include "index/Symbol.h"
#include "support/Threading.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/Utils.h"
#include "clang/Index/IndexSymbol.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <utility>

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
MATCHER_P(NumReferences, N, "") { return arg.References == N; }
MATCHER_P(hasOrign, O, "") { return bool(arg.Origin & O); }

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

TEST(FileSymbolsTest, UpdateAndGet) {
  FileSymbols FS;
  EXPECT_THAT(runFuzzyFind(*FS.buildIndex(IndexType::Light), ""), IsEmpty());

  FS.update("f1", numSlab(1, 3), refSlab(SymbolID("1"), "f1.cc"), nullptr,
            false);
  EXPECT_THAT(runFuzzyFind(*FS.buildIndex(IndexType::Light), ""),
              UnorderedElementsAre(QName("1"), QName("2"), QName("3")));
  EXPECT_THAT(getRefs(*FS.buildIndex(IndexType::Light), SymbolID("1")),
              RefsAre({FileURI("f1.cc")}));
}

TEST(FileSymbolsTest, Overlap) {
  FileSymbols FS;
  FS.update("f1", numSlab(1, 3), nullptr, nullptr, false);
  FS.update("f2", numSlab(3, 5), nullptr, nullptr, false);
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
            AllOf(QName("x"), DeclURI("file:///x1"), DefURI("file:///x2"))));
}

TEST(FileSymbolsTest, SnapshotAliveAfterRemove) {
  FileSymbols FS;

  SymbolID ID("1");
  FS.update("f1", numSlab(1, 3), refSlab(ID, "f1.cc"), nullptr, false);

  auto Symbols = FS.buildIndex(IndexType::Light);
  EXPECT_THAT(runFuzzyFind(*Symbols, ""),
              UnorderedElementsAre(QName("1"), QName("2"), QName("3")));
  EXPECT_THAT(getRefs(*Symbols, ID), RefsAre({FileURI("f1.cc")}));

  FS.update("f1", nullptr, nullptr, nullptr, false);
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
  File.HeaderCode = std::string(Code);
  auto AST = File.build();
  M.updatePreamble(testPath(File.Filename), /*Version=*/"null",
                   AST.getASTContext(), AST.getPreprocessorPtr(),
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
  buildPreamble(FooCpp, *CI, PI,
                /*StoreInMemory=*/true,
                [&](ASTContext &Ctx, std::shared_ptr<Preprocessor> PP,
                    const CanonicalIncludes &CanonIncludes) {
                  EXPECT_FALSE(IndexUpdated)
                      << "Expected only a single index update";
                  IndexUpdated = true;
                  Index.updatePreamble(FooCpp, /*Version=*/"null", Ctx,
                                       std::move(PP), CanonIncludes);
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
  Test.Code = std::string(MainCode.code());
  Test.Filename = "test.cc";
  auto AST = Test.build();
  Index.updateMain(Test.Filename, AST);
  // Add test2.cc
  TestTU Test2;
  Test2.HeaderCode = HeaderCode;
  Test2.Code = std::string(MainCode.code());
  Test2.Filename = "test2.cc";
  AST = Test2.build();
  Index.updateMain(Test2.Filename, AST);

  EXPECT_THAT(getRefs(Index, Foo.ID),
              RefsAre({AllOf(RefRange(MainCode.range("foo")),
                             FileURI("unittest:///test.cc")),
                       AllOf(RefRange(MainCode.range("foo")),
                             FileURI("unittest:///test2.cc"))}));
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
  Index.updateMain(Test.Filename, AST);

  auto HeaderMacro = findSymbol(Test.headerSymbols(), "HEADER_MACRO");
  EXPECT_THAT(getRefs(Index, HeaderMacro.ID),
              RefsAre({AllOf(RefRange(MainCode.range("ref1")),
                             FileURI("unittest:///test.cc"))}));

  auto MainFileMacro = findSymbol(Test.headerSymbols(), "MAINFILE_MACRO");
  EXPECT_THAT(getRefs(Index, MainFileMacro.ID),
              RefsAre({AllOf(RefRange(MainCode.range("def2")),
                             FileURI("unittest:///test.cc")),
                       AllOf(RefRange(MainCode.range("ref2")),
                             FileURI("unittest:///test.cc"))}));
}

TEST(FileIndexTest, CollectMacros) {
  FileIndex M;
  update(M, "f", "#define CLANGD 1");
  EXPECT_THAT(runFuzzyFind(M, ""), Contains(QName("CLANGD")));
}

TEST(FileIndexTest, Relations) {
  TestTU TU;
  TU.Filename = "f.cpp";
  TU.HeaderFilename = "f.h";
  TU.HeaderCode = "class A {}; class B : public A {};";
  auto AST = TU.build();
  FileIndex Index;
  Index.updatePreamble(testPath(TU.Filename), /*Version=*/"null",
                       AST.getASTContext(), AST.getPreprocessorPtr(),
                       AST.getCanonicalIncludes());
  SymbolID A = findSymbol(TU.headerSymbols(), "A").ID;
  uint32_t Results = 0;
  RelationsRequest Req;
  Req.Subjects.insert(A);
  Req.Predicate = RelationKind::BaseOf;
  Index.relations(Req, [&](const SymbolID &, const Symbol &) { ++Results; });
  EXPECT_EQ(Results, 1u);
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
              RefsAre({RefRange(Main.range())}));
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
  EXPECT_THAT(Symbols, ElementsAre(AllOf(DeclURI("unittest:///foo.h"),
                                         DefURI("unittest:///foo.cpp"),
                                         hasOrign(SymbolOrigin::Merge))));
}

TEST(FileSymbolsTest, CountReferencesNoRefSlabs) {
  FileSymbols FS;
  FS.update("f1", numSlab(1, 3), nullptr, nullptr, true);
  FS.update("f2", numSlab(1, 3), nullptr, nullptr, false);
  EXPECT_THAT(
      runFuzzyFind(*FS.buildIndex(IndexType::Light, DuplicateHandling::Merge),
                   ""),
      UnorderedElementsAre(AllOf(QName("1"), NumReferences(0u)),
                           AllOf(QName("2"), NumReferences(0u)),
                           AllOf(QName("3"), NumReferences(0u))));
}

TEST(FileSymbolsTest, CountReferencesWithRefSlabs) {
  FileSymbols FS;
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
      UnorderedElementsAre(AllOf(QName("1"), NumReferences(1u)),
                           AllOf(QName("2"), NumReferences(1u)),
                           AllOf(QName("3"), NumReferences(1u))));
}

TEST(FileIndexTest, StalePreambleSymbolsDeleted) {
  FileIndex M;
  TestTU File;
  File.HeaderFilename = "a.h";

  File.Filename = "f1.cpp";
  File.HeaderCode = "int a;";
  auto AST = File.build();
  M.updatePreamble(testPath(File.Filename), /*Version=*/"null",
                   AST.getASTContext(), AST.getPreprocessorPtr(),
                   AST.getCanonicalIncludes());
  EXPECT_THAT(runFuzzyFind(M, ""), UnorderedElementsAre(QName("a")));

  File.Filename = "f2.cpp";
  File.HeaderCode = "int b;";
  AST = File.build();
  M.updatePreamble(testPath(File.Filename), /*Version=*/"null",
                   AST.getASTContext(), AST.getPreprocessorPtr(),
                   AST.getCanonicalIncludes());
  EXPECT_THAT(runFuzzyFind(M, ""), UnorderedElementsAre(QName("b")));
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
    EXPECT_THAT(*Shard->Symbols, UnorderedElementsAre(QName("1")));
    EXPECT_THAT(*Shard->Refs, IsEmpty());
    EXPECT_THAT(
        *Shard->Relations,
        UnorderedElementsAre(Relation{Sym1.ID, RelationKind::BaseOf, Sym2.ID},
                             Relation{Sym2.ID, RelationKind::BaseOf, Sym1.ID},
                             Relation{Sym3.ID, RelationKind::BaseOf, Sym1.ID}));
    ASSERT_THAT(Shard->Sources->keys(), UnorderedElementsAre(AHeaderUri));
    EXPECT_THAT(Shard->Sources->lookup(AHeaderUri).DirectIncludes, IsEmpty());
    EXPECT_TRUE(Shard->Cmd.hasValue());
  }
  {
    auto Shard = ShardedIndex.getShard(BHeaderUri);
    ASSERT_TRUE(Shard);
    EXPECT_THAT(*Shard->Symbols, UnorderedElementsAre(QName("2")));
    EXPECT_THAT(*Shard->Refs, IsEmpty());
    EXPECT_THAT(
        *Shard->Relations,
        UnorderedElementsAre(Relation{Sym1.ID, RelationKind::BaseOf, Sym2.ID},
                             Relation{Sym2.ID, RelationKind::BaseOf, Sym1.ID}));
    ASSERT_THAT(Shard->Sources->keys(),
                UnorderedElementsAre(BHeaderUri, AHeaderUri));
    EXPECT_THAT(Shard->Sources->lookup(BHeaderUri).DirectIncludes,
                UnorderedElementsAre(AHeaderUri));
    EXPECT_TRUE(Shard->Cmd.hasValue());
  }
  {
    auto Shard = ShardedIndex.getShard(BSourceUri);
    ASSERT_TRUE(Shard);
    EXPECT_THAT(*Shard->Symbols, UnorderedElementsAre(QName("2")));
    EXPECT_THAT(*Shard->Refs, UnorderedElementsAre(Pair(Sym1.ID, _)));
    EXPECT_THAT(*Shard->Relations, IsEmpty());
    ASSERT_THAT(Shard->Sources->keys(),
                UnorderedElementsAre(BSourceUri, BHeaderUri));
    EXPECT_THAT(Shard->Sources->lookup(BSourceUri).DirectIncludes,
                UnorderedElementsAre(BHeaderUri));
    EXPECT_TRUE(Shard->Cmd.hasValue());
  }
}
} // namespace
} // namespace clangd
} // namespace clang
