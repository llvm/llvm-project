//===-- FileIndexTests.cpp  ---------------------------*- C++ -*-----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "index/FileIndex.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/PCHContainerOperations.h"
#include "clang/Frontend/Utils.h"
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

/// Create an ParsedAST for \p Code. Returns None if \p Code is empty.
llvm::Optional<ParsedAST> build(std::string Path, llvm::StringRef Code) {
  if (Code.empty())
    return llvm::None;
  const char *Args[] = {"clang", "-xc++", Path.c_str()};

  auto CI = createInvocationFromCommandLine(Args);

  auto Buf = llvm::MemoryBuffer::getMemBuffer(Code);
  auto AST = ParsedAST::Build(std::move(CI), nullptr, std::move(Buf),
                              std::make_shared<PCHContainerOperations>(),
                              vfs::getRealFileSystem());
  assert(AST.hasValue());
  return std::move(*AST);
}

TEST(FileIndexTest, IndexAST) {
  FileIndex M;
  M.update(
      "f1",
      build("f1", "namespace ns { void f() {} class X {}; }").getPointer());

  FuzzyFindRequest Req;
  Req.Query = "";
  Req.Scopes = {"ns::"};
  EXPECT_THAT(match(M, Req), UnorderedElementsAre("ns::f", "ns::X"));
}

TEST(FileIndexTest, NoLocal) {
  FileIndex M;
  M.update(
      "f1",
      build("f1", "namespace ns { void f() { int local = 0; } class X {}; }")
          .getPointer());

  FuzzyFindRequest Req;
  Req.Query = "";
  EXPECT_THAT(match(M, Req), UnorderedElementsAre("ns", "ns::f", "ns::X"));
}

TEST(FileIndexTest, IndexMultiASTAndDeduplicate) {
  FileIndex M;
  M.update(
      "f1",
      build("f1", "namespace ns { void f() {} class X {}; }").getPointer());
  M.update(
      "f2",
      build("f2", "namespace ns { void ff() {} class X {}; }").getPointer());

  FuzzyFindRequest Req;
  Req.Query = "";
  Req.Scopes = {"ns::"};
  EXPECT_THAT(match(M, Req), UnorderedElementsAre("ns::f", "ns::X", "ns::ff"));
}

TEST(FileIndexTest, RemoveAST) {
  FileIndex M;
  M.update(
      "f1",
      build("f1", "namespace ns { void f() {} class X {}; }").getPointer());

  FuzzyFindRequest Req;
  Req.Query = "";
  Req.Scopes = {"ns::"};
  EXPECT_THAT(match(M, Req), UnorderedElementsAre("ns::f", "ns::X"));

  M.update("f1", nullptr);
  EXPECT_THAT(match(M, Req), UnorderedElementsAre());
}

TEST(FileIndexTest, RemoveNonExisting) {
  FileIndex M;
  M.update("no", nullptr);
  EXPECT_THAT(match(M, FuzzyFindRequest()), UnorderedElementsAre());
}

TEST(FileIndexTest, IgnoreClassMembers) {
  FileIndex M;
  M.update("f1",
           build("f1", "class X { static int m1; int m2; static void f(); };")
               .getPointer());

  FuzzyFindRequest Req;
  Req.Query = "";
  EXPECT_THAT(match(M, Req), UnorderedElementsAre("X"));
}

} // namespace
} // namespace clangd
} // namespace clang
