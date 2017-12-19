//===-- IndexTests.cpp  -------------------------------*- C++ -*-----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "index/Index.h"
#include "index/MemIndex.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using testing::UnorderedElementsAre;

namespace clang {
namespace clangd {

namespace {

Symbol symbol(llvm::StringRef QName) {
  Symbol Sym;
  Sym.ID = SymbolID(QName.str());
  size_t Pos = QName.rfind("::");
  if (Pos == llvm::StringRef::npos) {
    Sym.Name = QName;
    Sym.Scope = "";
  } else {
    Sym.Name = QName.substr(Pos + 2);
    Sym.Scope = QName.substr(0, Pos);
  }
  return Sym;
}

struct SlabAndPointers {
  SymbolSlab Slab;
  std::vector<const Symbol *> Pointers;
};

// Create a slab of symbols with the given qualified names as both IDs and
// names. The life time of the slab is managed by the returned shared pointer.
// If \p WeakSymbols is provided, it will be pointed to the managed object in
// the returned shared pointer.
std::shared_ptr<std::vector<const Symbol *>>
generateSymbols(std::vector<std::string> QualifiedNames,
                std::weak_ptr<SlabAndPointers> *WeakSymbols = nullptr) {
  auto Slab = std::make_shared<SlabAndPointers>();
  if (WeakSymbols)
    *WeakSymbols = Slab;

  for (llvm::StringRef QName : QualifiedNames)
    Slab->Slab.insert(symbol(QName));

  for (const auto &Sym : Slab->Slab)
    Slab->Pointers.push_back(&Sym.second);

  auto *Pointers = &Slab->Pointers;
  return {std::move(Slab), Pointers};
}

// Create a slab of symbols with IDs and names [Begin, End], otherwise identical
// to the `generateSymbols` above.
std::shared_ptr<std::vector<const Symbol *>>
generateNumSymbols(int Begin, int End,
                   std::weak_ptr<SlabAndPointers> *WeakSymbols = nullptr) {
  std::vector<std::string> Names;
  for (int i = Begin; i <= End; i++)
    Names.push_back(std::to_string(i));
  return generateSymbols(Names, WeakSymbols);
}

std::vector<std::string> match(const SymbolIndex &I,
                               const FuzzyFindRequest &Req) {
  std::vector<std::string> Matches;
  auto Ctx = Context::empty();
  I.fuzzyFind(Ctx, Req, [&](const Symbol &Sym) {
    Matches.push_back(Sym.Scope + (Sym.Scope.empty() ? "" : "::") + Sym.Name);
  });
  return Matches;
}

TEST(MemIndexTest, MemIndexSymbolsRecycled) {
  MemIndex I;
  std::weak_ptr<SlabAndPointers> Symbols;
  I.build(generateNumSymbols(0, 10, &Symbols));
  FuzzyFindRequest Req;
  Req.Query = "7";
  EXPECT_THAT(match(I, Req), UnorderedElementsAre("7"));

  EXPECT_FALSE(Symbols.expired());
  // Release old symbols.
  I.build(generateNumSymbols(0, 0));
  EXPECT_TRUE(Symbols.expired());
}

TEST(MemIndexTest, MemIndexMatchSubstring) {
  MemIndex I;
  I.build(generateNumSymbols(5, 25));
  FuzzyFindRequest Req;
  Req.Query = "5";
  EXPECT_THAT(match(I, Req), UnorderedElementsAre("5", "15", "25"));
}

TEST(MemIndexTest, MemIndexDeduplicate) {
  auto Symbols = generateNumSymbols(0, 10);

  // Inject some duplicates and make sure we only match the same symbol once.
  auto Sym = symbol("7");
  Symbols->push_back(&Sym);
  Symbols->push_back(&Sym);
  Symbols->push_back(&Sym);

  FuzzyFindRequest Req;
  Req.Query = "7";
  MemIndex I;
  I.build(std::move(Symbols));
  auto Matches = match(I, Req);
  EXPECT_EQ(Matches.size(), 1u);
}

TEST(MemIndexTest, MemIndexLimitedNumMatches) {
  MemIndex I;
  I.build(generateNumSymbols(0, 100));
  FuzzyFindRequest Req;
  Req.Query = "5";
  Req.MaxCandidateCount = 3;
  auto Matches = match(I, Req);
  EXPECT_EQ(Matches.size(), Req.MaxCandidateCount);
}

TEST(MemIndexTest, MatchQualifiedNamesWithoutSpecificScope) {
  MemIndex I;
  I.build(generateSymbols({"a::xyz", "b::yz", "yz"}));
  FuzzyFindRequest Req;
  Req.Query = "y";
  auto Matches = match(I, Req);
  EXPECT_THAT(match(I, Req), UnorderedElementsAre("a::xyz", "b::yz", "yz"));
}

TEST(MemIndexTest, MatchQualifiedNamesWithGlobalScope) {
  MemIndex I;
  I.build(generateSymbols({"a::xyz", "b::yz", "yz"}));
  FuzzyFindRequest Req;
  Req.Query = "y";
  Req.Scopes.push_back("");
  auto Matches = match(I, Req);
  EXPECT_THAT(match(I, Req), UnorderedElementsAre("yz"));
}

TEST(MemIndexTest, MatchQualifiedNamesWithOneScope) {
  MemIndex I;
  I.build(generateSymbols({"a::xyz", "a::yy", "a::xz", "b::yz", "yz"}));
  FuzzyFindRequest Req;
  Req.Query = "y";
  Req.Scopes.push_back("a");
  auto Matches = match(I, Req);
  EXPECT_THAT(match(I, Req), UnorderedElementsAre("a::xyz", "a::yy"));
}

TEST(MemIndexTest, MatchQualifiedNamesWithMultipleScopes) {
  MemIndex I;
  I.build(generateSymbols({"a::xyz", "a::yy", "a::xz", "b::yz", "yz"}));
  FuzzyFindRequest Req;
  Req.Query = "y";
  Req.Scopes.push_back("a");
  Req.Scopes.push_back("b");
  auto Matches = match(I, Req);
  EXPECT_THAT(match(I, Req), UnorderedElementsAre("a::xyz", "a::yy", "b::yz"));
}

TEST(MemIndexTest, NoMatchNestedScopes) {
  MemIndex I;
  I.build(generateSymbols({"a::xyz", "a::b::yy"}));
  FuzzyFindRequest Req;
  Req.Query = "y";
  Req.Scopes.push_back("a");
  auto Matches = match(I, Req);
  EXPECT_THAT(match(I, Req), UnorderedElementsAre("a::xyz"));
}

} // namespace
} // namespace clangd
} // namespace clang
