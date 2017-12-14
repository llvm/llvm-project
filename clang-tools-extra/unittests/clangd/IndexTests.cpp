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

Symbol symbol(llvm::StringRef ID) {
  Symbol Sym;
  Sym.ID = SymbolID(ID);
  Sym.QualifiedName = ID;
  return Sym;
}

struct SlabAndPointers {
  SymbolSlab Slab;
  std::vector<const Symbol *> Pointers;
};

// Create a slab of symbols with IDs and names [Begin, End]. The life time of
// the slab is managed by the returned shared pointer. If \p WeakSymbols is
// provided, it will be pointed to the managed object in the returned shared
// pointer.
std::shared_ptr<std::vector<const Symbol *>>
generateNumSymbols(int Begin, int End,
                   std::weak_ptr<SlabAndPointers> *WeakSymbols = nullptr) {
  auto Slab = std::make_shared<SlabAndPointers>();
  if (WeakSymbols)
    *WeakSymbols = Slab;

  for (int i = Begin; i <= End; i++)
    Slab->Slab.insert(symbol(std::to_string(i)));

  for (const auto &Sym : Slab->Slab)
    Slab->Pointers.push_back(&Sym.second);

  auto *Pointers = &Slab->Pointers;
  return {std::move(Slab), Pointers};
}

std::vector<std::string> match(const SymbolIndex &I,
                               const FuzzyFindRequest &Req) {
  std::vector<std::string> Matches;
  auto Ctx = Context::empty();
  I.fuzzyFind(Ctx, Req,
              [&](const Symbol &Sym) { Matches.push_back(Sym.QualifiedName); });
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

} // namespace
} // namespace clangd
} // namespace clang
