//===-- FileIndexTests.cpp  ---------------------------*- C++ -*-----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ClangdUnit.h"
#include "TestFS.h"
#include "TestTU.h"
#include "index/FileIndex.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/PCHContainerOperations.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using testing::UnorderedElementsAre;

namespace clang {
namespace clangd {

namespace {

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

std::vector<std::string>
getSymbolNames(const std::vector<const Symbol *> &Symbols) {
  std::vector<std::string> Names;
  for (const Symbol *Sym : Symbols)
    Names.push_back(Sym->Name);
  return Names;
}

TEST(FileSymbolsTest, UpdateAndGet) {
  FileSymbols FS;
  EXPECT_THAT(getSymbolNames(*FS.allSymbols()), UnorderedElementsAre());

  FS.update("f1", numSlab(1, 3));
  EXPECT_THAT(getSymbolNames(*FS.allSymbols()),
              UnorderedElementsAre("1", "2", "3"));
}

TEST(FileSymbolsTest, Overlap) {
  FileSymbols FS;
  FS.update("f1", numSlab(1, 3));
  FS.update("f2", numSlab(3, 5));
  EXPECT_THAT(getSymbolNames(*FS.allSymbols()),
              UnorderedElementsAre("1", "2", "3", "3", "4", "5"));
}

TEST(FileSymbolsTest, SnapshotAliveAfterRemove) {
  FileSymbols FS;

  FS.update("f1", numSlab(1, 3));

  auto Symbols = FS.allSymbols();
  EXPECT_THAT(getSymbolNames(*Symbols), UnorderedElementsAre("1", "2", "3"));

  FS.update("f1", nullptr);
  EXPECT_THAT(getSymbolNames(*FS.allSymbols()), UnorderedElementsAre());
  EXPECT_THAT(getSymbolNames(*Symbols), UnorderedElementsAre("1", "2", "3"));
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
  M.update(File.Filename, &AST.getASTContext(), AST.getPreprocessorPtr());
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

TEST(FileIndexTest, RemoveAST) {
  FileIndex M;
  update(M, "f1", "namespace ns { void f() {} class X {}; }");

  FuzzyFindRequest Req;
  Req.Query = "";
  Req.Scopes = {"ns::"};
  EXPECT_THAT(match(M, Req), UnorderedElementsAre("ns::f", "ns::X"));

  M.update("f1.cpp", nullptr, nullptr);
  EXPECT_THAT(match(M, Req), UnorderedElementsAre());
}

TEST(FileIndexTest, RemoveNonExisting) {
  FileIndex M;
  M.update("no.cpp", nullptr, nullptr);
  EXPECT_THAT(match(M, FuzzyFindRequest()), UnorderedElementsAre());
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
    EXPECT_TRUE(Sym.Detail->IncludeHeader.empty());
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
      [&Index, &IndexUpdated](PathRef FilePath, ASTContext &Ctx,
                              std::shared_ptr<Preprocessor> PP) {
        EXPECT_FALSE(IndexUpdated) << "Expected only a single index update";
        IndexUpdated = true;
        Index.update(FilePath, &Ctx, std::move(PP));
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

} // namespace
} // namespace clangd
} // namespace clang
