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
#include "index/Merge.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using testing::UnorderedElementsAre;
using testing::Pointee;

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
    Sym.Scope = QName.substr(0, Pos + 2);
  }
  return Sym;
}

MATCHER_P(Named, N, "") { return arg.Name == N; }

TEST(SymbolSlab, FindAndIterate) {
  SymbolSlab::Builder B;
  B.insert(symbol("Z"));
  B.insert(symbol("Y"));
  B.insert(symbol("X"));
  EXPECT_EQ(nullptr, B.find(SymbolID("W")));
  for (const char *Sym : {"X", "Y", "Z"})
    EXPECT_THAT(B.find(SymbolID(Sym)), Pointee(Named(Sym)));

  SymbolSlab S = std::move(B).build();
  EXPECT_THAT(S, UnorderedElementsAre(Named("X"), Named("Y"), Named("Z")));
  EXPECT_EQ(S.end(), S.find(SymbolID("W")));
  for (const char *Sym : {"X", "Y", "Z"})
    EXPECT_THAT(*S.find(SymbolID(Sym)), Named(Sym));
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
  SymbolSlab::Builder Slab;
  for (llvm::StringRef QName : QualifiedNames)
    Slab.insert(symbol(QName));

  auto Storage = std::make_shared<SlabAndPointers>();
  Storage->Slab = std::move(Slab).build();
  for (const auto &Sym : Storage->Slab)
    Storage->Pointers.push_back(&Sym);
  if (WeakSymbols)
    *WeakSymbols = Storage;
  auto *Pointers = &Storage->Pointers;
  return {std::move(Storage), Pointers};
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

std::string getQualifiedName(const Symbol &Sym) {
  return (Sym.Scope + Sym.Name).str();
}

std::vector<std::string> match(const SymbolIndex &I,
                               const FuzzyFindRequest &Req,
                               bool *Incomplete = nullptr) {
  std::vector<std::string> Matches;
  bool IsIncomplete = I.fuzzyFind(Req, [&](const Symbol &Sym) {
    Matches.push_back(getQualifiedName(Sym));
  });
  if (Incomplete)
    *Incomplete = IsIncomplete;
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
  bool Incomplete;
  auto Matches = match(I, Req, &Incomplete);
  EXPECT_EQ(Matches.size(), Req.MaxCandidateCount);
  EXPECT_TRUE(Incomplete);
}

TEST(MemIndexTest, FuzzyMatch) {
  MemIndex I;
  I.build(
      generateSymbols({"LaughingOutLoud", "LionPopulation", "LittleOldLady"}));
  FuzzyFindRequest Req;
  Req.Query = "lol";
  Req.MaxCandidateCount = 2;
  EXPECT_THAT(match(I, Req),
              UnorderedElementsAre("LaughingOutLoud", "LittleOldLady"));
}

TEST(MemIndexTest, MatchQualifiedNamesWithoutSpecificScope) {
  MemIndex I;
  I.build(generateSymbols({"a::y1", "b::y2", "y3"}));
  FuzzyFindRequest Req;
  Req.Query = "y";
  EXPECT_THAT(match(I, Req), UnorderedElementsAre("a::y1", "b::y2", "y3"));
}

TEST(MemIndexTest, MatchQualifiedNamesWithGlobalScope) {
  MemIndex I;
  I.build(generateSymbols({"a::y1", "b::y2", "y3"}));
  FuzzyFindRequest Req;
  Req.Query = "y";
  Req.Scopes = {""};
  EXPECT_THAT(match(I, Req), UnorderedElementsAre("y3"));
}

TEST(MemIndexTest, MatchQualifiedNamesWithOneScope) {
  MemIndex I;
  I.build(generateSymbols({"a::y1", "a::y2", "a::x", "b::y2", "y3"}));
  FuzzyFindRequest Req;
  Req.Query = "y";
  Req.Scopes = {"a::"};
  EXPECT_THAT(match(I, Req), UnorderedElementsAre("a::y1", "a::y2"));
}

TEST(MemIndexTest, MatchQualifiedNamesWithMultipleScopes) {
  MemIndex I;
  I.build(generateSymbols({"a::y1", "a::y2", "a::x", "b::y3", "y3"}));
  FuzzyFindRequest Req;
  Req.Query = "y";
  Req.Scopes = {"a::", "b::"};
  EXPECT_THAT(match(I, Req), UnorderedElementsAre("a::y1", "a::y2", "b::y3"));
}

TEST(MemIndexTest, NoMatchNestedScopes) {
  MemIndex I;
  I.build(generateSymbols({"a::y1", "a::b::y2"}));
  FuzzyFindRequest Req;
  Req.Query = "y";
  Req.Scopes = {"a::"};
  EXPECT_THAT(match(I, Req), UnorderedElementsAre("a::y1"));
}

TEST(MemIndexTest, IgnoreCases) {
  MemIndex I;
  I.build(generateSymbols({"ns::ABC", "ns::abc"}));
  FuzzyFindRequest Req;
  Req.Query = "AB";
  Req.Scopes = {"ns::"};
  EXPECT_THAT(match(I, Req), UnorderedElementsAre("ns::ABC", "ns::abc"));
}

// Returns qualified names of symbols with any of IDs in the index.
std::vector<std::string> lookup(const SymbolIndex &I,
                                llvm::ArrayRef<SymbolID> IDs) {
  LookupRequest Req;
  Req.IDs.insert(IDs.begin(), IDs.end());
  std::vector<std::string> Results;
  I.lookup(Req, [&](const Symbol &Sym) {
    Results.push_back(getQualifiedName(Sym));
  });
  return Results;
}

TEST(MemIndexTest, Lookup) {
  MemIndex I;
  I.build(generateSymbols({"ns::abc", "ns::xyz"}));
  EXPECT_THAT(lookup(I, SymbolID("ns::abc")), UnorderedElementsAre("ns::abc"));
  EXPECT_THAT(lookup(I, {SymbolID("ns::abc"), SymbolID("ns::xyz")}),
              UnorderedElementsAre("ns::abc", "ns::xyz"));
  EXPECT_THAT(lookup(I, {SymbolID("ns::nonono"), SymbolID("ns::xyz")}),
              UnorderedElementsAre("ns::xyz"));
  EXPECT_THAT(lookup(I, SymbolID("ns::nonono")), UnorderedElementsAre());
}

TEST(MergeIndexTest, Lookup) {
  MemIndex I, J;
  I.build(generateSymbols({"ns::A", "ns::B"}));
  J.build(generateSymbols({"ns::B", "ns::C"}));
  EXPECT_THAT(lookup(*mergeIndex(&I, &J), SymbolID("ns::A")),
              UnorderedElementsAre("ns::A"));
  EXPECT_THAT(lookup(*mergeIndex(&I, &J), SymbolID("ns::B")),
              UnorderedElementsAre("ns::B"));
  EXPECT_THAT(lookup(*mergeIndex(&I, &J), SymbolID("ns::C")),
              UnorderedElementsAre("ns::C"));
  EXPECT_THAT(
      lookup(*mergeIndex(&I, &J), {SymbolID("ns::A"), SymbolID("ns::B")}),
      UnorderedElementsAre("ns::A", "ns::B"));
  EXPECT_THAT(
      lookup(*mergeIndex(&I, &J), {SymbolID("ns::A"), SymbolID("ns::C")}),
      UnorderedElementsAre("ns::A", "ns::C"));
  EXPECT_THAT(lookup(*mergeIndex(&I, &J), SymbolID("ns::D")),
              UnorderedElementsAre());
  EXPECT_THAT(lookup(*mergeIndex(&I, &J), {}), UnorderedElementsAre());
}

TEST(MergeIndexTest, FuzzyFind) {
  MemIndex I, J;
  I.build(generateSymbols({"ns::A", "ns::B"}));
  J.build(generateSymbols({"ns::B", "ns::C"}));
  FuzzyFindRequest Req;
  Req.Scopes = {"ns::"};
  EXPECT_THAT(match(*mergeIndex(&I, &J), Req),
              UnorderedElementsAre("ns::A", "ns::B", "ns::C"));
}

TEST(MergeTest, Merge) {
  Symbol L, R;
  L.ID = R.ID = SymbolID("hello");
  L.Name = R.Name = "Foo";                    // same in both
  L.CanonicalDeclaration.FileURI = "file:///left.h"; // differs
  R.CanonicalDeclaration.FileURI = "file:///right.h";
  L.References = 1;
  R.References = 2;
  L.Signature = "()";                   // present in left only
  R.CompletionSnippetSuffix = "{$1:0}"; // present in right only
  Symbol::Details DetL, DetR;
  DetL.ReturnType = "DetL";
  DetR.ReturnType = "DetR";
  DetR.Documentation = "--doc--";
  L.Detail = &DetL;
  R.Detail = &DetR;
  L.Origin = SymbolOrigin::Dynamic;
  R.Origin = SymbolOrigin::Static;

  Symbol::Details Scratch;
  Symbol M = mergeSymbol(L, R, &Scratch);
  EXPECT_EQ(M.Name, "Foo");
  EXPECT_EQ(M.CanonicalDeclaration.FileURI, "file:///left.h");
  EXPECT_EQ(M.References, 3u);
  EXPECT_EQ(M.Signature, "()");
  EXPECT_EQ(M.CompletionSnippetSuffix, "{$1:0}");
  ASSERT_TRUE(M.Detail);
  EXPECT_EQ(M.Detail->ReturnType, "DetL");
  EXPECT_EQ(M.Detail->Documentation, "--doc--");
  EXPECT_EQ(M.Origin,
            SymbolOrigin::Dynamic | SymbolOrigin::Static | SymbolOrigin::Merge);
}

TEST(MergeTest, PreferSymbolWithDefn) {
  Symbol L, R;
  Symbol::Details Scratch;

  L.ID = R.ID = SymbolID("hello");
  L.CanonicalDeclaration.FileURI = "file:/left.h";
  R.CanonicalDeclaration.FileURI = "file:/right.h";
  L.Name = "left";
  R.Name = "right";

  Symbol M = mergeSymbol(L, R, &Scratch);
  EXPECT_EQ(M.CanonicalDeclaration.FileURI, "file:/left.h");
  EXPECT_EQ(M.Definition.FileURI, "");
  EXPECT_EQ(M.Name, "left");

  R.Definition.FileURI = "file:/right.cpp"; // Now right will be favored.
  M = mergeSymbol(L, R, &Scratch);
  EXPECT_EQ(M.CanonicalDeclaration.FileURI, "file:/right.h");
  EXPECT_EQ(M.Definition.FileURI, "file:/right.cpp");
  EXPECT_EQ(M.Name, "right");
}

} // namespace
} // namespace clangd
} // namespace clang
