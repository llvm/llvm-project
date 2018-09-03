//===-- IndexTests.cpp  -------------------------------*- C++ -*-----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "TestIndex.h"
#include "TestTU.h"
#include "gmock/gmock.h"
#include "index/FileIndex.h"
#include "index/Index.h"
#include "index/MemIndex.h"
#include "index/Merge.h"
#include "gtest/gtest.h"

using testing::Pointee;
using testing::UnorderedElementsAre;
using testing::AllOf;

namespace clang {
namespace clangd {
namespace {

std::shared_ptr<MemIndex::OccurrenceMap> emptyOccurrences() {
  return llvm::make_unique<MemIndex::OccurrenceMap>();
}

MATCHER_P(Named, N, "") { return arg.Name == N; }
MATCHER_P(OccurrenceRange, Range, "") {
  return std::tie(arg.Location.Start.Line, arg.Location.Start.Column,
                  arg.Location.End.Line, arg.Location.End.Column) ==
         std::tie(Range.start.line, Range.start.character, Range.end.line,
                  Range.end.character);
}
MATCHER_P(FileURI, F, "") { return arg.Location.FileURI == F; }

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

TEST(MemIndexTest, MemIndexSymbolsRecycled) {
  MemIndex I;
  std::weak_ptr<SlabAndPointers> Symbols;
  I.build(generateNumSymbols(0, 10, &Symbols), emptyOccurrences());
  FuzzyFindRequest Req;
  Req.Query = "7";
  EXPECT_THAT(match(I, Req), UnorderedElementsAre("7"));

  EXPECT_FALSE(Symbols.expired());
  // Release old symbols.
  I.build(generateNumSymbols(0, 0), emptyOccurrences());
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
  I.build(std::move(Symbols), emptyOccurrences());
  auto Matches = match(I, Req);
  EXPECT_EQ(Matches.size(), 1u);
}

TEST(MemIndexTest, MemIndexLimitedNumMatches) {
  MemIndex I;
  I.build(generateNumSymbols(0, 100), emptyOccurrences());
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
      generateSymbols({"LaughingOutLoud", "LionPopulation", "LittleOldLady"}),
      emptyOccurrences());
  FuzzyFindRequest Req;
  Req.Query = "lol";
  Req.MaxCandidateCount = 2;
  EXPECT_THAT(match(I, Req),
              UnorderedElementsAre("LaughingOutLoud", "LittleOldLady"));
}

TEST(MemIndexTest, MatchQualifiedNamesWithoutSpecificScope) {
  MemIndex I;
  I.build(generateSymbols({"a::y1", "b::y2", "y3"}), emptyOccurrences());
  FuzzyFindRequest Req;
  Req.Query = "y";
  EXPECT_THAT(match(I, Req), UnorderedElementsAre("a::y1", "b::y2", "y3"));
}

TEST(MemIndexTest, MatchQualifiedNamesWithGlobalScope) {
  MemIndex I;
  I.build(generateSymbols({"a::y1", "b::y2", "y3"}), emptyOccurrences());
  FuzzyFindRequest Req;
  Req.Query = "y";
  Req.Scopes = {""};
  EXPECT_THAT(match(I, Req), UnorderedElementsAre("y3"));
}

TEST(MemIndexTest, MatchQualifiedNamesWithOneScope) {
  MemIndex I;
  I.build(generateSymbols({"a::y1", "a::y2", "a::x", "b::y2", "y3"}),
          emptyOccurrences());
  FuzzyFindRequest Req;
  Req.Query = "y";
  Req.Scopes = {"a::"};
  EXPECT_THAT(match(I, Req), UnorderedElementsAre("a::y1", "a::y2"));
}

TEST(MemIndexTest, MatchQualifiedNamesWithMultipleScopes) {
  MemIndex I;
  I.build(generateSymbols({"a::y1", "a::y2", "a::x", "b::y3", "y3"}),
          emptyOccurrences());
  FuzzyFindRequest Req;
  Req.Query = "y";
  Req.Scopes = {"a::", "b::"};
  EXPECT_THAT(match(I, Req), UnorderedElementsAre("a::y1", "a::y2", "b::y3"));
}

TEST(MemIndexTest, NoMatchNestedScopes) {
  MemIndex I;
  I.build(generateSymbols({"a::y1", "a::b::y2"}), emptyOccurrences());
  FuzzyFindRequest Req;
  Req.Query = "y";
  Req.Scopes = {"a::"};
  EXPECT_THAT(match(I, Req), UnorderedElementsAre("a::y1"));
}

TEST(MemIndexTest, IgnoreCases) {
  MemIndex I;
  I.build(generateSymbols({"ns::ABC", "ns::abc"}), emptyOccurrences());
  FuzzyFindRequest Req;
  Req.Query = "AB";
  Req.Scopes = {"ns::"};
  EXPECT_THAT(match(I, Req), UnorderedElementsAre("ns::ABC", "ns::abc"));
}

TEST(MemIndexTest, Lookup) {
  MemIndex I;
  I.build(generateSymbols({"ns::abc", "ns::xyz"}), emptyOccurrences());
  EXPECT_THAT(lookup(I, SymbolID("ns::abc")), UnorderedElementsAre("ns::abc"));
  EXPECT_THAT(lookup(I, {SymbolID("ns::abc"), SymbolID("ns::xyz")}),
              UnorderedElementsAre("ns::abc", "ns::xyz"));
  EXPECT_THAT(lookup(I, {SymbolID("ns::nonono"), SymbolID("ns::xyz")}),
              UnorderedElementsAre("ns::xyz"));
  EXPECT_THAT(lookup(I, SymbolID("ns::nonono")), UnorderedElementsAre());
}

TEST(MergeIndexTest, Lookup) {
  MemIndex I, J;
  I.build(generateSymbols({"ns::A", "ns::B"}), emptyOccurrences());
  J.build(generateSymbols({"ns::B", "ns::C"}), emptyOccurrences());
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
  I.build(generateSymbols({"ns::A", "ns::B"}), emptyOccurrences());
  J.build(generateSymbols({"ns::B", "ns::C"}), emptyOccurrences());
  FuzzyFindRequest Req;
  Req.Scopes = {"ns::"};
  EXPECT_THAT(match(*mergeIndex(&I, &J), Req),
              UnorderedElementsAre("ns::A", "ns::B", "ns::C"));
}

TEST(MergeTest, Merge) {
  Symbol L, R;
  L.ID = R.ID = SymbolID("hello");
  L.Name = R.Name = "Foo";                           // same in both
  L.CanonicalDeclaration.FileURI = "file:///left.h"; // differs
  R.CanonicalDeclaration.FileURI = "file:///right.h";
  L.References = 1;
  R.References = 2;
  L.Signature = "()";                   // present in left only
  R.CompletionSnippetSuffix = "{$1:0}"; // present in right only
  R.Documentation = "--doc--";
  L.Origin = SymbolOrigin::Dynamic;
  R.Origin = SymbolOrigin::Static;

  Symbol M = mergeSymbol(L, R);
  EXPECT_EQ(M.Name, "Foo");
  EXPECT_EQ(M.CanonicalDeclaration.FileURI, "file:///left.h");
  EXPECT_EQ(M.References, 3u);
  EXPECT_EQ(M.Signature, "()");
  EXPECT_EQ(M.CompletionSnippetSuffix, "{$1:0}");
  EXPECT_EQ(M.Documentation, "--doc--");
  EXPECT_EQ(M.Origin,
            SymbolOrigin::Dynamic | SymbolOrigin::Static | SymbolOrigin::Merge);
}

TEST(MergeTest, PreferSymbolWithDefn) {
  Symbol L, R;

  L.ID = R.ID = SymbolID("hello");
  L.CanonicalDeclaration.FileURI = "file:/left.h";
  R.CanonicalDeclaration.FileURI = "file:/right.h";
  L.Name = "left";
  R.Name = "right";

  Symbol M = mergeSymbol(L, R);
  EXPECT_EQ(M.CanonicalDeclaration.FileURI, "file:/left.h");
  EXPECT_EQ(M.Definition.FileURI, "");
  EXPECT_EQ(M.Name, "left");

  R.Definition.FileURI = "file:/right.cpp"; // Now right will be favored.
  M = mergeSymbol(L, R);
  EXPECT_EQ(M.CanonicalDeclaration.FileURI, "file:/right.h");
  EXPECT_EQ(M.Definition.FileURI, "file:/right.cpp");
  EXPECT_EQ(M.Name, "right");
}

TEST(MergeIndexTest, FindOccurrences) {
  FileIndex Dyn({"unittest"});
  FileIndex StaticIndex({"unittest"});
  auto MergedIndex = mergeIndex(&Dyn, &StaticIndex);

  const char *HeaderCode = "class Foo;";
  auto HeaderSymbols = TestTU::withHeaderCode("class Foo;").headerSymbols();
  auto Foo = findSymbol(HeaderSymbols, "Foo");

  // Build dynamic index for test.cc.
  Annotations Test1Code(R"(class $Foo[[Foo]];)");
  TestTU Test;
  Test.HeaderCode = HeaderCode;
  Test.Code = Test1Code.code();
  Test.Filename = "test.cc";
  auto AST = Test.build();
  Dyn.update(Test.Filename, &AST.getASTContext(), AST.getPreprocessorPtr(),
             AST.getLocalTopLevelDecls());

  // Build static index for test.cc.
  Test.HeaderCode = HeaderCode;
  Test.Code = "// static\nclass Foo {};";
  Test.Filename = "test.cc";
  auto StaticAST = Test.build();
  // Add stale occurrences for test.cc.
  StaticIndex.update(Test.Filename, &StaticAST.getASTContext(),
                     StaticAST.getPreprocessorPtr(),
                     StaticAST.getLocalTopLevelDecls());

  // Add occcurrences for test2.cc
  Annotations Test2Code(R"(class $Foo[[Foo]] {};)");
  TestTU Test2;
  Test2.HeaderCode = HeaderCode;
  Test2.Code = Test2Code.code();
  Test2.Filename = "test2.cc";
  StaticAST = Test2.build();
  StaticIndex.update(Test2.Filename, &StaticAST.getASTContext(),
                     StaticAST.getPreprocessorPtr(),
                     StaticAST.getLocalTopLevelDecls());

  OccurrencesRequest Request;
  Request.IDs = {Foo.ID};
  Request.Filter = AllOccurrenceKinds;
  std::vector<SymbolOccurrence> Results;
  MergedIndex->findOccurrences(
      Request, [&](const SymbolOccurrence &O) { Results.push_back(O); });

  EXPECT_THAT(Results, UnorderedElementsAre(
                           AllOf(OccurrenceRange(Test1Code.range("Foo")),
                                 FileURI("unittest:///test.cc")),
                           AllOf(OccurrenceRange(Test2Code.range("Foo")),
                                 FileURI("unittest:///test2.cc"))));
}

MATCHER_P2(IncludeHeaderWithRef, IncludeHeader, References,  "") {
  return (arg.IncludeHeader == IncludeHeader) && (arg.References == References);
}

TEST(MergeTest, MergeIncludesOnDifferentDefinitions) {
  Symbol L, R;
  L.Name = "left";
  R.Name = "right";
  L.ID = R.ID = SymbolID("hello");
  L.IncludeHeaders.emplace_back("common", 1);
  R.IncludeHeaders.emplace_back("common", 1);
  R.IncludeHeaders.emplace_back("new", 1);

  // Both have no definition.
  Symbol M = mergeSymbol(L, R);
  EXPECT_THAT(M.IncludeHeaders,
              UnorderedElementsAre(IncludeHeaderWithRef("common", 2u),
                                   IncludeHeaderWithRef("new", 1u)));

  // Only merge references of the same includes but do not merge new #includes.
  L.Definition.FileURI = "file:/left.h";
  M = mergeSymbol(L, R);
  EXPECT_THAT(M.IncludeHeaders,
              UnorderedElementsAre(IncludeHeaderWithRef("common", 2u)));

  // Definitions are the same.
  R.Definition.FileURI = "file:/right.h";
  M = mergeSymbol(L, R);
  EXPECT_THAT(M.IncludeHeaders,
              UnorderedElementsAre(IncludeHeaderWithRef("common", 2u),
                                   IncludeHeaderWithRef("new", 1u)));

  // Definitions are different.
  R.Definition.FileURI = "file:/right.h";
  M = mergeSymbol(L, R);
  EXPECT_THAT(M.IncludeHeaders,
              UnorderedElementsAre(IncludeHeaderWithRef("common", 2u),
                                   IncludeHeaderWithRef("new", 1u)));
}

} // namespace
} // namespace clangd
} // namespace clang
