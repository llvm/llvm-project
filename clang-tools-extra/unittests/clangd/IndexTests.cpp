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
#include "index/FileIndex.h"
#include "index/Index.h"
#include "index/MemIndex.h"
#include "index/Merge.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using testing::_;
using testing::AllOf;
using testing::ElementsAre;
using testing::Pair;
using testing::Pointee;
using testing::UnorderedElementsAre;
using namespace llvm;

namespace clang {
namespace clangd {
namespace {

MATCHER_P(Named, N, "") { return arg.Name == N; }
MATCHER_P(RefRange, Range, "") {
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

TEST(SwapIndexTest, OldIndexRecycled) {
  auto Token = std::make_shared<int>();
  std::weak_ptr<int> WeakToken = Token;

  SwapIndex S(llvm::make_unique<MemIndex>(
      SymbolSlab(), RefSlab(), std::move(Token), /*BackingDataSize=*/0));
  EXPECT_FALSE(WeakToken.expired());      // Current MemIndex keeps it alive.
  S.reset(llvm::make_unique<MemIndex>()); // Now the MemIndex is destroyed.
  EXPECT_TRUE(WeakToken.expired());       // So the token is too.
}

TEST(MemIndexTest, MemIndexDeduplicate) {
  std::vector<Symbol> Symbols = {symbol("1"), symbol("2"), symbol("3"),
                                 symbol("2") /* duplicate */};
  FuzzyFindRequest Req;
  Req.Query = "2";
  MemIndex I(Symbols, RefSlab());
  EXPECT_THAT(match(I, Req), ElementsAre("2"));
}

TEST(MemIndexTest, MemIndexLimitedNumMatches) {
  auto I = MemIndex::build(generateNumSymbols(0, 100), RefSlab());
  FuzzyFindRequest Req;
  Req.Query = "5";
  Req.Limit = 3;
  bool Incomplete;
  auto Matches = match(*I, Req, &Incomplete);
  EXPECT_TRUE(Req.Limit);
  EXPECT_EQ(Matches.size(), *Req.Limit);
  EXPECT_TRUE(Incomplete);
}

TEST(MemIndexTest, FuzzyMatch) {
  auto I = MemIndex::build(
      generateSymbols({"LaughingOutLoud", "LionPopulation", "LittleOldLady"}),
      RefSlab());
  FuzzyFindRequest Req;
  Req.Query = "lol";
  Req.Limit = 2;
  EXPECT_THAT(match(*I, Req),
              UnorderedElementsAre("LaughingOutLoud", "LittleOldLady"));
}

TEST(MemIndexTest, MatchQualifiedNamesWithoutSpecificScope) {
  auto I =
      MemIndex::build(generateSymbols({"a::y1", "b::y2", "y3"}), RefSlab());
  FuzzyFindRequest Req;
  Req.Query = "y";
  EXPECT_THAT(match(*I, Req), UnorderedElementsAre("a::y1", "b::y2", "y3"));
}

TEST(MemIndexTest, MatchQualifiedNamesWithGlobalScope) {
  auto I =
      MemIndex::build(generateSymbols({"a::y1", "b::y2", "y3"}), RefSlab());
  FuzzyFindRequest Req;
  Req.Query = "y";
  Req.Scopes = {""};
  EXPECT_THAT(match(*I, Req), UnorderedElementsAre("y3"));
}

TEST(MemIndexTest, MatchQualifiedNamesWithOneScope) {
  auto I = MemIndex::build(
      generateSymbols({"a::y1", "a::y2", "a::x", "b::y2", "y3"}), RefSlab());
  FuzzyFindRequest Req;
  Req.Query = "y";
  Req.Scopes = {"a::"};
  EXPECT_THAT(match(*I, Req), UnorderedElementsAre("a::y1", "a::y2"));
}

TEST(MemIndexTest, MatchQualifiedNamesWithMultipleScopes) {
  auto I = MemIndex::build(
      generateSymbols({"a::y1", "a::y2", "a::x", "b::y3", "y3"}), RefSlab());
  FuzzyFindRequest Req;
  Req.Query = "y";
  Req.Scopes = {"a::", "b::"};
  EXPECT_THAT(match(*I, Req), UnorderedElementsAre("a::y1", "a::y2", "b::y3"));
}

TEST(MemIndexTest, NoMatchNestedScopes) {
  auto I = MemIndex::build(generateSymbols({"a::y1", "a::b::y2"}), RefSlab());
  FuzzyFindRequest Req;
  Req.Query = "y";
  Req.Scopes = {"a::"};
  EXPECT_THAT(match(*I, Req), UnorderedElementsAre("a::y1"));
}

TEST(MemIndexTest, IgnoreCases) {
  auto I = MemIndex::build(generateSymbols({"ns::ABC", "ns::abc"}), RefSlab());
  FuzzyFindRequest Req;
  Req.Query = "AB";
  Req.Scopes = {"ns::"};
  EXPECT_THAT(match(*I, Req), UnorderedElementsAre("ns::ABC", "ns::abc"));
}

TEST(MemIndexTest, Lookup) {
  auto I = MemIndex::build(generateSymbols({"ns::abc", "ns::xyz"}), RefSlab());
  EXPECT_THAT(lookup(*I, SymbolID("ns::abc")), UnorderedElementsAre("ns::abc"));
  EXPECT_THAT(lookup(*I, {SymbolID("ns::abc"), SymbolID("ns::xyz")}),
              UnorderedElementsAre("ns::abc", "ns::xyz"));
  EXPECT_THAT(lookup(*I, {SymbolID("ns::nonono"), SymbolID("ns::xyz")}),
              UnorderedElementsAre("ns::xyz"));
  EXPECT_THAT(lookup(*I, SymbolID("ns::nonono")), UnorderedElementsAre());
}

TEST(MergeIndexTest, Lookup) {
  auto I = MemIndex::build(generateSymbols({"ns::A", "ns::B"}), RefSlab()),
       J = MemIndex::build(generateSymbols({"ns::B", "ns::C"}), RefSlab());
  auto M = mergeIndex(I.get(), J.get());
  EXPECT_THAT(lookup(*M, SymbolID("ns::A")), UnorderedElementsAre("ns::A"));
  EXPECT_THAT(lookup(*M, SymbolID("ns::B")), UnorderedElementsAre("ns::B"));
  EXPECT_THAT(lookup(*M, SymbolID("ns::C")), UnorderedElementsAre("ns::C"));
  EXPECT_THAT(lookup(*M, {SymbolID("ns::A"), SymbolID("ns::B")}),
              UnorderedElementsAre("ns::A", "ns::B"));
  EXPECT_THAT(lookup(*M, {SymbolID("ns::A"), SymbolID("ns::C")}),
              UnorderedElementsAre("ns::A", "ns::C"));
  EXPECT_THAT(lookup(*M, SymbolID("ns::D")), UnorderedElementsAre());
  EXPECT_THAT(lookup(*M, {}), UnorderedElementsAre());
}

TEST(MergeIndexTest, FuzzyFind) {
  auto I = MemIndex::build(generateSymbols({"ns::A", "ns::B"}), RefSlab()),
       J = MemIndex::build(generateSymbols({"ns::B", "ns::C"}), RefSlab());
  FuzzyFindRequest Req;
  Req.Scopes = {"ns::"};
  EXPECT_THAT(match(*mergeIndex(I.get(), J.get()), Req),
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

TEST(MergeIndexTest, Refs) {
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
  // Add stale refs for test.cc.
  StaticIndex.update(Test.Filename, &StaticAST.getASTContext(),
                     StaticAST.getPreprocessorPtr(),
                     StaticAST.getLocalTopLevelDecls());

  // Add refs for test2.cc
  Annotations Test2Code(R"(class $Foo[[Foo]] {};)");
  TestTU Test2;
  Test2.HeaderCode = HeaderCode;
  Test2.Code = Test2Code.code();
  Test2.Filename = "test2.cc";
  StaticAST = Test2.build();
  StaticIndex.update(Test2.Filename, &StaticAST.getASTContext(),
                     StaticAST.getPreprocessorPtr(),
                     StaticAST.getLocalTopLevelDecls());

  RefsRequest Request;
  Request.IDs = {Foo.ID};
  RefSlab::Builder Results;
  MergedIndex->refs(Request, [&](const Ref &O) { Results.insert(Foo.ID, O); });

  EXPECT_THAT(
      std::move(Results).build(),
      ElementsAre(Pair(
          _, UnorderedElementsAre(AllOf(RefRange(Test1Code.range("Foo")),
                                        FileURI("unittest:///test.cc")),
                                  AllOf(RefRange(Test2Code.range("Foo")),
                                        FileURI("unittest:///test2.cc"))))));
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
