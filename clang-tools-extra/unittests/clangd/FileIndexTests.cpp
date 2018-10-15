//===-- FileIndexTests.cpp  ---------------------------*- C++ -*-----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "AST.h"
#include "ClangdUnit.h"
#include "TestFS.h"
#include "TestTU.h"
#include "gmock/gmock.h"
#include "index/FileIndex.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/PCHContainerOperations.h"
#include "clang/Frontend/Utils.h"
#include "clang/Index/IndexSymbol.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "gtest/gtest.h"

using testing::_;
using testing::AllOf;
using testing::ElementsAre;
using testing::Pair;
using testing::UnorderedElementsAre;

MATCHER_P(RefRange, Range, "") {
  return std::tie(arg.Location.Start.Line, arg.Location.Start.Column,
                  arg.Location.End.Line, arg.Location.End.Column) ==
         std::tie(Range.start.line, Range.start.character, Range.end.line,
                  Range.end.character);
}
MATCHER_P(FileURI, F, "") { return arg.Location.FileURI == F; }

namespace clang {
namespace clangd {
namespace {
testing::Matcher<const RefSlab &>
RefsAre(std::vector<testing::Matcher<Ref>> Matchers) {
  return ElementsAre(testing::Pair(_, UnorderedElementsAreArray(Matchers)));
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

std::unique_ptr<RefSlab> refSlab(const SymbolID &ID, llvm::StringRef Path) {
  RefSlab::Builder Slab;
  Ref R;
  R.Location.FileURI = Path;
  R.Kind = RefKind::Reference;
  Slab.insert(ID, R);
  return llvm::make_unique<RefSlab>(std::move(Slab).build());
}

std::vector<std::string> getSymbolNames(const SymbolIndex &I,
                                        std::string Query = "") {
  FuzzyFindRequest Req;
  Req.Query = Query;
  std::vector<std::string> Names;
  I.fuzzyFind(Req, [&](const Symbol &S) { Names.push_back(S.Name); });
  return Names;
}

RefSlab getRefs(const SymbolIndex &I, SymbolID ID) {
  RefsRequest Req;
  Req.IDs = {ID};
  RefSlab::Builder Slab;
  I.refs(Req, [&](const Ref &S) { Slab.insert(ID, S); });
  return std::move(Slab).build();
}

TEST(FileSymbolsTest, UpdateAndGet) {
  FileSymbols FS;
  EXPECT_THAT(getSymbolNames(*FS.buildMemIndex()), UnorderedElementsAre());

  FS.update("f1", numSlab(1, 3), refSlab(SymbolID("1"), "f1.cc"));
  EXPECT_THAT(getSymbolNames(*FS.buildMemIndex()),
              UnorderedElementsAre("1", "2", "3"));
  EXPECT_THAT(getRefs(*FS.buildMemIndex(), SymbolID("1")),
              RefsAre({FileURI("f1.cc")}));
}

TEST(FileSymbolsTest, Overlap) {
  FileSymbols FS;
  FS.update("f1", numSlab(1, 3), nullptr);
  FS.update("f2", numSlab(3, 5), nullptr);
  EXPECT_THAT(getSymbolNames(*FS.buildMemIndex()),
              UnorderedElementsAre("1", "2", "3", "4", "5"));
}

TEST(FileSymbolsTest, SnapshotAliveAfterRemove) {
  FileSymbols FS;

  SymbolID ID("1");
  FS.update("f1", numSlab(1, 3), refSlab(ID, "f1.cc"));

  auto Symbols = FS.buildMemIndex();
  EXPECT_THAT(getSymbolNames(*Symbols), UnorderedElementsAre("1", "2", "3"));
  EXPECT_THAT(getRefs(*Symbols, ID), RefsAre({FileURI("f1.cc")}));

  FS.update("f1", nullptr, nullptr);
  auto Empty = FS.buildMemIndex();
  EXPECT_THAT(getSymbolNames(*Empty), UnorderedElementsAre());
  EXPECT_THAT(getRefs(*Empty, ID), ElementsAre());

  EXPECT_THAT(getSymbolNames(*Symbols), UnorderedElementsAre("1", "2", "3"));
  EXPECT_THAT(getRefs(*Symbols, ID), RefsAre({FileURI("f1.cc")}));
}

std::vector<std::string> match(const SymbolIndex &I,
                               const FuzzyFindRequest &Req) {
  std::vector<std::string> Matches;
  I.fuzzyFind(Req, [&](const Symbol &Sym) {
    Matches.push_back((Sym.Scope + Sym.Name).str());
  });
  return Matches;
}

// Adds Basename.cpp, which includes Basename.h, which contains Code.
void update(FileIndex &M, llvm::StringRef Basename, llvm::StringRef Code) {
  TestTU File;
  File.Filename = (Basename + ".cpp").str();
  File.HeaderFilename = (Basename + ".h").str();
  File.HeaderCode = Code;
  auto AST = File.build();
  M.updatePreamble(File.Filename, AST.getASTContext(),
                   AST.getPreprocessorPtr());
}

TEST(FileIndexTest, CustomizedURIScheme) {
  FileIndex M({"unittest"});
  update(M, "f", "class string {};");

  FuzzyFindRequest Req;
  Req.Query = "";
  bool SeenSymbol = false;
  M.fuzzyFind(Req, [&](const Symbol &Sym) {
    EXPECT_EQ(Sym.CanonicalDeclaration.FileURI, "unittest:///f.h");
    SeenSymbol = true;
  });
  EXPECT_TRUE(SeenSymbol);
}

TEST(FileIndexTest, IndexAST) {
  FileIndex M;
  update(M, "f1", "namespace ns { void f() {} class X {}; }");

  FuzzyFindRequest Req;
  Req.Query = "";
  Req.Scopes = {"ns::"};
  EXPECT_THAT(match(M, Req), UnorderedElementsAre("ns::f", "ns::X"));
}

TEST(FileIndexTest, NoLocal) {
  FileIndex M;
  update(M, "f1", "namespace ns { void f() { int local = 0; } class X {}; }");

  FuzzyFindRequest Req;
  Req.Query = "";
  EXPECT_THAT(match(M, Req), UnorderedElementsAre("ns", "ns::f", "ns::X"));
}

TEST(FileIndexTest, IndexMultiASTAndDeduplicate) {
  FileIndex M;
  update(M, "f1", "namespace ns { void f() {} class X {}; }");
  update(M, "f2", "namespace ns { void ff() {} class X {}; }");

  FuzzyFindRequest Req;
  Req.Query = "";
  Req.Scopes = {"ns::"};
  EXPECT_THAT(match(M, Req), UnorderedElementsAre("ns::f", "ns::X", "ns::ff"));
}

TEST(FileIndexTest, ClassMembers) {
  FileIndex M;
  update(M, "f1", "class X { static int m1; int m2; static void f(); };");

  FuzzyFindRequest Req;
  Req.Query = "";
  EXPECT_THAT(match(M, Req),
              UnorderedElementsAre("X", "X::m1", "X::m2", "X::f"));
}

TEST(FileIndexTest, NoIncludeCollected) {
  FileIndex M;
  update(M, "f", "class string {};");

  FuzzyFindRequest Req;
  Req.Query = "";
  bool SeenSymbol = false;
  M.fuzzyFind(Req, [&](const Symbol &Sym) {
    EXPECT_TRUE(Sym.IncludeHeaders.empty());
    SeenSymbol = true;
  });
  EXPECT_TRUE(SeenSymbol);
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

  FuzzyFindRequest Req;
  Req.Query = "";
  bool SeenVector = false;
  bool SeenMakeVector = false;
  M.fuzzyFind(Req, [&](const Symbol &Sym) {
    if (Sym.Name == "vector") {
      EXPECT_EQ(Sym.Signature, "<class Ty>");
      EXPECT_EQ(Sym.CompletionSnippetSuffix, "<${1:class Ty}>");
      SeenVector = true;
      return;
    }

    if (Sym.Name == "make_vector") {
      EXPECT_EQ(Sym.Signature, "<class Ty>(Arg A)");
      EXPECT_EQ(Sym.CompletionSnippetSuffix, "<${1:class Ty}>(${2:Arg A})");
      SeenMakeVector = true;
    }
  });
  EXPECT_TRUE(SeenVector);
  EXPECT_TRUE(SeenMakeVector);
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
      std::make_shared<PCHContainerOperations>(), /*StoreInMemory=*/true,
      [&](ASTContext &Ctx, std::shared_ptr<Preprocessor> PP) {
        EXPECT_FALSE(IndexUpdated) << "Expected only a single index update";
        IndexUpdated = true;
        Index.updatePreamble(FooCpp, Ctx, std::move(PP));
      });
  ASSERT_TRUE(IndexUpdated);

  // Check the index contains symbols from the preamble, but not from the main
  // file.
  FuzzyFindRequest Req;
  Req.Query = "";
  Req.Scopes = {"", "ns_in_header::"};

  EXPECT_THAT(
      match(Index, Req),
      UnorderedElementsAre("ns_in_header", "ns_in_header::func_in_header"));
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

  FileIndex Index(/*URISchemes*/ {"unittest"});
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

  FuzzyFindRequest Req;
  Req.Query = "";
  bool SeenSymbol = false;
  M.fuzzyFind(Req, [&](const Symbol &Sym) {
    EXPECT_EQ(Sym.Name, "CLANGD");
    EXPECT_EQ(Sym.SymInfo.Kind, index::SymbolKind::Macro);
    SeenSymbol = true;
  });
  EXPECT_TRUE(SeenSymbol);
}

TEST(FileIndexTest, ReferencesInMainFileWithPreamble) {
  const std::string Header = R"cpp(
    class Foo {};
  )cpp";
  Annotations Main(R"cpp(
    #include "foo.h"
    void f() {
      [[Foo]] foo;
    }
  )cpp");
  auto MainFile = testPath("foo.cpp");
  auto HeaderFile = testPath("foo.h");
  std::vector<const char*> Cmd = {"clang", "-xc++", MainFile.c_str()};
  // Preparse ParseInputs.
  ParseInputs PI;
  PI.CompileCommand.Directory = testRoot();
  PI.CompileCommand.Filename = MainFile;
  PI.CompileCommand.CommandLine = {Cmd.begin(), Cmd.end()};
  PI.Contents = Main.code();
  PI.FS = buildTestFS({{MainFile, Main.code()}, {HeaderFile, Header}});

  // Prepare preamble.
  auto CI = buildCompilerInvocation(PI);
  auto PreambleData = buildPreamble(
      MainFile,
      *buildCompilerInvocation(PI), /*OldPreamble=*/nullptr,
      tooling::CompileCommand(), PI,
      std::make_shared<PCHContainerOperations>(), /*StoreInMemory=*/true,
      [&](ASTContext &Ctx, std::shared_ptr<Preprocessor> PP) {});
  // Build AST for main file with preamble.
  auto AST = ParsedAST::build(
      createInvocationFromCommandLine(Cmd), PreambleData,
      llvm::MemoryBuffer::getMemBufferCopy(Main.code()),
      std::make_shared<PCHContainerOperations>(),
      PI.FS);
  ASSERT_TRUE(AST);
  FileIndex Index;
  Index.updateMain(MainFile, *AST);

  auto Foo =
      findSymbol(TestTU::withHeaderCode(Header).headerSymbols(), "Foo");
  RefsRequest Request;
  Request.IDs.insert(Foo.ID);

  // Expect to see references in main file, references in headers are excluded
  // because we only index main AST.
  EXPECT_THAT(getRefs(Index, Foo.ID), RefsAre({RefRange(Main.range())}));
}

} // namespace
} // namespace clangd
} // namespace clang
