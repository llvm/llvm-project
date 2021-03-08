//===-- IndexTests.cpp  -------------------------------*- C++ -*-----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "TestIndex.h"
#include "TestTU.h"
#include "index/FileIndex.h"
#include "index/Index.h"
#include "index/MemIndex.h"
#include "index/Merge.h"
#include "index/Symbol.h"
#include "clang/Index/IndexSymbol.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ::testing::_;
using ::testing::AllOf;
using ::testing::AnyOf;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::Pointee;
using ::testing::UnorderedElementsAre;

namespace clang {
namespace clangd {
namespace {

MATCHER_P(Named, N, "") { return arg.Name == N; }
MATCHER_P(RefRange, Range, "") {
  return std::make_tuple(arg.Location.Start.line(), arg.Location.Start.column(),
                         arg.Location.End.line(), arg.Location.End.column()) ==
         std::make_tuple(Range.start.line, Range.start.character,
                         Range.end.line, Range.end.character);
}
MATCHER_P(FileURI, F, "") { return StringRef(arg.Location.FileURI) == F; }

TEST(SymbolLocation, Position) {
  using Position = SymbolLocation::Position;
  Position Pos;

  Pos.setLine(1);
  EXPECT_EQ(1u, Pos.line());
  Pos.setColumn(2);
  EXPECT_EQ(2u, Pos.column());
  EXPECT_FALSE(Pos.hasOverflow());

  Pos.setLine(Position::MaxLine + 1); // overflow
  EXPECT_TRUE(Pos.hasOverflow());
  EXPECT_EQ(Pos.line(), Position::MaxLine);
  Pos.setLine(1); // reset the overflowed line.

  Pos.setColumn(Position::MaxColumn + 1); // overflow
  EXPECT_TRUE(Pos.hasOverflow());
  EXPECT_EQ(Pos.column(), Position::MaxColumn);
}

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

TEST(RelationSlab, Lookup) {
  SymbolID A{"A"};
  SymbolID B{"B"};
  SymbolID C{"C"};
  SymbolID D{"D"};

  RelationSlab::Builder Builder;
  Builder.insert(Relation{A, RelationKind::BaseOf, B});
  Builder.insert(Relation{A, RelationKind::BaseOf, C});
  Builder.insert(Relation{B, RelationKind::BaseOf, D});
  Builder.insert(Relation{C, RelationKind::BaseOf, D});

  RelationSlab Slab = std::move(Builder).build();
  EXPECT_THAT(Slab.lookup(A, RelationKind::BaseOf),
              UnorderedElementsAre(Relation{A, RelationKind::BaseOf, B},
                                   Relation{A, RelationKind::BaseOf, C}));
}

TEST(RelationSlab, Duplicates) {
  SymbolID A{"A"};
  SymbolID B{"B"};
  SymbolID C{"C"};

  RelationSlab::Builder Builder;
  Builder.insert(Relation{A, RelationKind::BaseOf, B});
  Builder.insert(Relation{A, RelationKind::BaseOf, C});
  Builder.insert(Relation{A, RelationKind::BaseOf, B});

  RelationSlab Slab = std::move(Builder).build();
  EXPECT_THAT(Slab, UnorderedElementsAre(Relation{A, RelationKind::BaseOf, B},
                                         Relation{A, RelationKind::BaseOf, C}));
}

TEST(SwapIndexTest, OldIndexRecycled) {
  auto Token = std::make_shared<int>();
  std::weak_ptr<int> WeakToken = Token;

  SwapIndex S(std::make_unique<MemIndex>(SymbolSlab(), RefSlab(),
                                          RelationSlab(), std::move(Token),
                                          /*BackingDataSize=*/0));
  EXPECT_FALSE(WeakToken.expired());      // Current MemIndex keeps it alive.
  S.reset(std::make_unique<MemIndex>()); // Now the MemIndex is destroyed.
  EXPECT_TRUE(WeakToken.expired());       // So the token is too.
}

TEST(MemIndexTest, MemIndexDeduplicate) {
  std::vector<Symbol> Symbols = {symbol("1"), symbol("2"), symbol("3"),
                                 symbol("2") /* duplicate */};
  FuzzyFindRequest Req;
  Req.Query = "2";
  Req.AnyScope = true;
  MemIndex I(Symbols, RefSlab(), RelationSlab());
  EXPECT_THAT(match(I, Req), ElementsAre("2"));
}

TEST(MemIndexTest, MemIndexLimitedNumMatches) {
  auto I =
      MemIndex::build(generateNumSymbols(0, 100), RefSlab(), RelationSlab());
  FuzzyFindRequest Req;
  Req.Query = "5";
  Req.AnyScope = true;
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
      RefSlab(), RelationSlab());
  FuzzyFindRequest Req;
  Req.Query = "lol";
  Req.AnyScope = true;
  Req.Limit = 2;
  EXPECT_THAT(match(*I, Req),
              UnorderedElementsAre("LaughingOutLoud", "LittleOldLady"));
}

TEST(MemIndexTest, MatchQualifiedNamesWithoutSpecificScope) {
  auto I = MemIndex::build(generateSymbols({"a::y1", "b::y2", "y3"}), RefSlab(),
                           RelationSlab());
  FuzzyFindRequest Req;
  Req.Query = "y";
  Req.AnyScope = true;
  EXPECT_THAT(match(*I, Req), UnorderedElementsAre("a::y1", "b::y2", "y3"));
}

TEST(MemIndexTest, MatchQualifiedNamesWithGlobalScope) {
  auto I = MemIndex::build(generateSymbols({"a::y1", "b::y2", "y3"}), RefSlab(),
                           RelationSlab());
  FuzzyFindRequest Req;
  Req.Query = "y";
  Req.Scopes = {""};
  EXPECT_THAT(match(*I, Req), UnorderedElementsAre("y3"));
}

TEST(MemIndexTest, MatchQualifiedNamesWithOneScope) {
  auto I = MemIndex::build(
      generateSymbols({"a::y1", "a::y2", "a::x", "b::y2", "y3"}), RefSlab(),
      RelationSlab());
  FuzzyFindRequest Req;
  Req.Query = "y";
  Req.Scopes = {"a::"};
  EXPECT_THAT(match(*I, Req), UnorderedElementsAre("a::y1", "a::y2"));
}

TEST(MemIndexTest, MatchQualifiedNamesWithMultipleScopes) {
  auto I = MemIndex::build(
      generateSymbols({"a::y1", "a::y2", "a::x", "b::y3", "y3"}), RefSlab(),
      RelationSlab());
  FuzzyFindRequest Req;
  Req.Query = "y";
  Req.Scopes = {"a::", "b::"};
  EXPECT_THAT(match(*I, Req), UnorderedElementsAre("a::y1", "a::y2", "b::y3"));
}

TEST(MemIndexTest, NoMatchNestedScopes) {
  auto I = MemIndex::build(generateSymbols({"a::y1", "a::b::y2"}), RefSlab(),
                           RelationSlab());
  FuzzyFindRequest Req;
  Req.Query = "y";
  Req.Scopes = {"a::"};
  EXPECT_THAT(match(*I, Req), UnorderedElementsAre("a::y1"));
}

TEST(MemIndexTest, IgnoreCases) {
  auto I = MemIndex::build(generateSymbols({"ns::ABC", "ns::abc"}), RefSlab(),
                           RelationSlab());
  FuzzyFindRequest Req;
  Req.Query = "AB";
  Req.Scopes = {"ns::"};
  EXPECT_THAT(match(*I, Req), UnorderedElementsAre("ns::ABC", "ns::abc"));
}

TEST(MemIndexTest, Lookup) {
  auto I = MemIndex::build(generateSymbols({"ns::abc", "ns::xyz"}), RefSlab(),
                           RelationSlab());
  EXPECT_THAT(lookup(*I, SymbolID("ns::abc")), UnorderedElementsAre("ns::abc"));
  EXPECT_THAT(lookup(*I, {SymbolID("ns::abc"), SymbolID("ns::xyz")}),
              UnorderedElementsAre("ns::abc", "ns::xyz"));
  EXPECT_THAT(lookup(*I, {SymbolID("ns::nonono"), SymbolID("ns::xyz")}),
              UnorderedElementsAre("ns::xyz"));
  EXPECT_THAT(lookup(*I, SymbolID("ns::nonono")), UnorderedElementsAre());
}

TEST(MemIndexTest, IndexedFiles) {
  SymbolSlab Symbols;
  RefSlab Refs;
  auto Size = Symbols.bytes() + Refs.bytes();
  auto Data = std::make_pair(std::move(Symbols), std::move(Refs));
  llvm::StringSet<> Files = {"unittest:///foo.cc", "unittest:///bar.cc"};
  MemIndex I(std::move(Data.first), std::move(Data.second), RelationSlab(),
             std::move(Files), IndexContents::All, std::move(Data), Size);
  auto ContainsFile = I.indexedFiles();
  EXPECT_EQ(ContainsFile("unittest:///foo.cc"), IndexContents::All);
  EXPECT_EQ(ContainsFile("unittest:///bar.cc"), IndexContents::All);
  EXPECT_EQ(ContainsFile("unittest:///foobar.cc"), IndexContents::None);
}

TEST(MemIndexTest, TemplateSpecialization) {
  SymbolSlab::Builder B;

  Symbol S = symbol("TempSpec");
  S.ID = SymbolID("1");
  B.insert(S);

  S = symbol("TempSpec");
  S.ID = SymbolID("2");
  S.TemplateSpecializationArgs = "<int, bool>";
  S.SymInfo.Properties = static_cast<index::SymbolPropertySet>(
      index::SymbolProperty::TemplateSpecialization);
  B.insert(S);

  S = symbol("TempSpec");
  S.ID = SymbolID("3");
  S.TemplateSpecializationArgs = "<int, U>";
  S.SymInfo.Properties = static_cast<index::SymbolPropertySet>(
      index::SymbolProperty::TemplatePartialSpecialization);
  B.insert(S);

  auto I = MemIndex::build(std::move(B).build(), RefSlab(), RelationSlab());
  FuzzyFindRequest Req;
  Req.AnyScope = true;

  Req.Query = "TempSpec";
  EXPECT_THAT(match(*I, Req),
              UnorderedElementsAre("TempSpec", "TempSpec<int, bool>",
                                   "TempSpec<int, U>"));

  // FIXME: Add filtering for template argument list.
  Req.Query = "TempSpec<int";
  EXPECT_THAT(match(*I, Req), IsEmpty());
}

TEST(MergeIndexTest, Lookup) {
  auto I = MemIndex::build(generateSymbols({"ns::A", "ns::B"}), RefSlab(),
                           RelationSlab()),
       J = MemIndex::build(generateSymbols({"ns::B", "ns::C"}), RefSlab(),
                           RelationSlab());
  MergedIndex M(I.get(), J.get());
  EXPECT_THAT(lookup(M, SymbolID("ns::A")), UnorderedElementsAre("ns::A"));
  EXPECT_THAT(lookup(M, SymbolID("ns::B")), UnorderedElementsAre("ns::B"));
  EXPECT_THAT(lookup(M, SymbolID("ns::C")), UnorderedElementsAre("ns::C"));
  EXPECT_THAT(lookup(M, {SymbolID("ns::A"), SymbolID("ns::B")}),
              UnorderedElementsAre("ns::A", "ns::B"));
  EXPECT_THAT(lookup(M, {SymbolID("ns::A"), SymbolID("ns::C")}),
              UnorderedElementsAre("ns::A", "ns::C"));
  EXPECT_THAT(lookup(M, SymbolID("ns::D")), UnorderedElementsAre());
  EXPECT_THAT(lookup(M, {}), UnorderedElementsAre());
}

TEST(MergeIndexTest, LookupRemovedDefinition) {
  FileIndex DynamicIndex, StaticIndex;
  MergedIndex Merge(&DynamicIndex, &StaticIndex);

  const char *HeaderCode = "class Foo;";
  auto HeaderSymbols = TestTU::withHeaderCode(HeaderCode).headerSymbols();
  auto Foo = findSymbol(HeaderSymbols, "Foo");

  // Build static index for test.cc with Foo definition
  TestTU Test;
  Test.HeaderCode = HeaderCode;
  Test.Code = "class Foo {};";
  Test.Filename = "test.cc";
  auto AST = Test.build();
  StaticIndex.updateMain(testPath(Test.Filename), AST);

  // Remove Foo definition from test.cc, i.e. build dynamic index for test.cc
  // without Foo definition.
  Test.Code = "class Foo;";
  AST = Test.build();
  DynamicIndex.updateMain(testPath(Test.Filename), AST);

  // Merged index should not return the symbol definition if this definition
  // location is inside a file from the dynamic index.
  LookupRequest LookupReq;
  LookupReq.IDs = {Foo.ID};
  unsigned SymbolCounter = 0;
  Merge.lookup(LookupReq, [&](const Symbol &Sym) {
    ++SymbolCounter;
    EXPECT_FALSE(Sym.Definition);
  });
  EXPECT_EQ(SymbolCounter, 1u);
}

TEST(MergeIndexTest, FuzzyFind) {
  auto I = MemIndex::build(generateSymbols({"ns::A", "ns::B"}), RefSlab(),
                           RelationSlab()),
       J = MemIndex::build(generateSymbols({"ns::B", "ns::C"}), RefSlab(),
                           RelationSlab());
  FuzzyFindRequest Req;
  Req.Scopes = {"ns::"};
  EXPECT_THAT(match(MergedIndex(I.get(), J.get()), Req),
              UnorderedElementsAre("ns::A", "ns::B", "ns::C"));
}

TEST(MergeIndexTest, FuzzyFindRemovedSymbol) {
  FileIndex DynamicIndex, StaticIndex;
  MergedIndex Merge(&DynamicIndex, &StaticIndex);

  const char *HeaderCode = "class Foo;";
  auto HeaderSymbols = TestTU::withHeaderCode(HeaderCode).headerSymbols();
  auto Foo = findSymbol(HeaderSymbols, "Foo");

  // Build static index for test.cc with Foo symbol
  TestTU Test;
  Test.HeaderCode = HeaderCode;
  Test.Code = "class Foo {};";
  Test.Filename = "test.cc";
  auto AST = Test.build();
  StaticIndex.updateMain(testPath(Test.Filename), AST);

  // Remove Foo symbol, i.e. build dynamic index for test.cc, which is empty.
  Test.HeaderCode = "";
  Test.Code = "";
  AST = Test.build();
  DynamicIndex.updateMain(testPath(Test.Filename), AST);

  // Merged index should not return removed symbol.
  FuzzyFindRequest Req;
  Req.AnyScope = true;
  Req.Query = "Foo";
  unsigned SymbolCounter = 0;
  bool IsIncomplete =
      Merge.fuzzyFind(Req, [&](const Symbol &) { ++SymbolCounter; });
  EXPECT_FALSE(IsIncomplete);
  EXPECT_EQ(SymbolCounter, 0u);
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
  R.Type = "expectedType";

  Symbol M = mergeSymbol(L, R);
  EXPECT_EQ(M.Name, "Foo");
  EXPECT_EQ(StringRef(M.CanonicalDeclaration.FileURI), "file:///left.h");
  EXPECT_EQ(M.References, 3u);
  EXPECT_EQ(M.Signature, "()");
  EXPECT_EQ(M.CompletionSnippetSuffix, "{$1:0}");
  EXPECT_EQ(M.Documentation, "--doc--");
  EXPECT_EQ(M.Type, "expectedType");
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
  EXPECT_EQ(StringRef(M.CanonicalDeclaration.FileURI), "file:/left.h");
  EXPECT_EQ(StringRef(M.Definition.FileURI), "");
  EXPECT_EQ(M.Name, "left");

  R.Definition.FileURI = "file:/right.cpp"; // Now right will be favored.
  M = mergeSymbol(L, R);
  EXPECT_EQ(StringRef(M.CanonicalDeclaration.FileURI), "file:/right.h");
  EXPECT_EQ(StringRef(M.Definition.FileURI), "file:/right.cpp");
  EXPECT_EQ(M.Name, "right");
}

TEST(MergeTest, PreferSymbolLocationInCodegenFile) {
  Symbol L, R;

  L.ID = R.ID = SymbolID("hello");
  L.CanonicalDeclaration.FileURI = "file:/x.proto.h";
  R.CanonicalDeclaration.FileURI = "file:/x.proto";

  Symbol M = mergeSymbol(L, R);
  EXPECT_EQ(StringRef(M.CanonicalDeclaration.FileURI), "file:/x.proto");

  // Prefer L if both have codegen suffix.
  L.CanonicalDeclaration.FileURI = "file:/y.proto";
  M = mergeSymbol(L, R);
  EXPECT_EQ(StringRef(M.CanonicalDeclaration.FileURI), "file:/y.proto");
}

TEST(MergeIndexTest, Refs) {
  FileIndex Dyn;
  FileIndex StaticIndex;
  MergedIndex Merge(&Dyn, &StaticIndex);

  const char *HeaderCode = "class Foo;";
  auto HeaderSymbols = TestTU::withHeaderCode("class Foo;").headerSymbols();
  auto Foo = findSymbol(HeaderSymbols, "Foo");

  // Build dynamic index for test.cc.
  Annotations Test1Code(R"(class $Foo[[Foo]];)");
  TestTU Test;
  Test.HeaderCode = HeaderCode;
  Test.Code = std::string(Test1Code.code());
  Test.Filename = "test.cc";
  auto AST = Test.build();
  Dyn.updateMain(testPath(Test.Filename), AST);

  // Build static index for test.cc.
  Test.HeaderCode = HeaderCode;
  Test.Code = "// static\nclass Foo {};";
  Test.Filename = "test.cc";
  auto StaticAST = Test.build();
  // Add stale refs for test.cc.
  StaticIndex.updateMain(testPath(Test.Filename), StaticAST);

  // Add refs for test2.cc
  Annotations Test2Code(R"(class $Foo[[Foo]] {};)");
  TestTU Test2;
  Test2.HeaderCode = HeaderCode;
  Test2.Code = std::string(Test2Code.code());
  Test2.Filename = "test2.cc";
  StaticAST = Test2.build();
  StaticIndex.updateMain(testPath(Test2.Filename), StaticAST);

  RefsRequest Request;
  Request.IDs = {Foo.ID};
  RefSlab::Builder Results;
  EXPECT_FALSE(
      Merge.refs(Request, [&](const Ref &O) { Results.insert(Foo.ID, O); }));
  EXPECT_THAT(
      std::move(Results).build(),
      ElementsAre(Pair(
          _, UnorderedElementsAre(AllOf(RefRange(Test1Code.range("Foo")),
                                        FileURI("unittest:///test.cc")),
                                  AllOf(RefRange(Test2Code.range("Foo")),
                                        FileURI("unittest:///test2.cc"))))));

  Request.Limit = 1;
  RefSlab::Builder Results2;
  EXPECT_TRUE(
      Merge.refs(Request, [&](const Ref &O) { Results2.insert(Foo.ID, O); }));

  // Remove all refs for test.cc from dynamic index,
  // merged index should not return results from static index for test.cc.
  Test.Code = "";
  AST = Test.build();
  Dyn.updateMain(testPath(Test.Filename), AST);

  Request.Limit = llvm::None;
  RefSlab::Builder Results3;
  EXPECT_FALSE(
      Merge.refs(Request, [&](const Ref &O) { Results3.insert(Foo.ID, O); }));
  EXPECT_THAT(std::move(Results3).build(),
              ElementsAre(Pair(_, UnorderedElementsAre(AllOf(
                                      RefRange(Test2Code.range("Foo")),
                                      FileURI("unittest:///test2.cc"))))));
}

TEST(MergeIndexTest, IndexedFiles) {
  SymbolSlab DynSymbols;
  RefSlab DynRefs;
  auto DynSize = DynSymbols.bytes() + DynRefs.bytes();
  auto DynData = std::make_pair(std::move(DynSymbols), std::move(DynRefs));
  llvm::StringSet<> DynFiles = {"unittest:///foo.cc"};
  MemIndex DynIndex(std::move(DynData.first), std::move(DynData.second),
                    RelationSlab(), std::move(DynFiles), IndexContents::Symbols,
                    std::move(DynData), DynSize);
  SymbolSlab StaticSymbols;
  RefSlab StaticRefs;
  auto StaticData =
      std::make_pair(std::move(StaticSymbols), std::move(StaticRefs));
  llvm::StringSet<> StaticFiles = {"unittest:///foo.cc", "unittest:///bar.cc"};
  MemIndex StaticIndex(
      std::move(StaticData.first), std::move(StaticData.second), RelationSlab(),
      std::move(StaticFiles), IndexContents::References, std::move(StaticData),
      StaticSymbols.bytes() + StaticRefs.bytes());
  MergedIndex Merge(&DynIndex, &StaticIndex);

  auto ContainsFile = Merge.indexedFiles();
  EXPECT_EQ(ContainsFile("unittest:///foo.cc"),
            IndexContents::Symbols | IndexContents::References);
  EXPECT_EQ(ContainsFile("unittest:///bar.cc"), IndexContents::References);
  EXPECT_EQ(ContainsFile("unittest:///foobar.cc"), IndexContents::None);
}

TEST(MergeIndexTest, NonDocumentation) {
  using index::SymbolKind;
  Symbol L, R;
  L.ID = R.ID = SymbolID("x");
  L.Definition.FileURI = "file:/x.h";
  R.Documentation = "Forward declarations because x.h is too big to include";
  for (auto ClassLikeKind :
       {SymbolKind::Class, SymbolKind::Struct, SymbolKind::Union}) {
    L.SymInfo.Kind = ClassLikeKind;
    EXPECT_EQ(mergeSymbol(L, R).Documentation, "");
  }

  L.SymInfo.Kind = SymbolKind::Function;
  R.Documentation = "Documentation from non-class symbols should be included";
  EXPECT_EQ(mergeSymbol(L, R).Documentation, R.Documentation);
}

MATCHER_P2(IncludeHeaderWithRef, IncludeHeader, References, "") {
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
