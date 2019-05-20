//===-- SerializationTests.cpp - Binary and YAML serialization unit tests -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "index/Index.h"
#include "index/Serialization.h"
#include "llvm/Support/SHA1.h"
#include "llvm/Support/ScopedPrinter.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using testing::_;
using testing::AllOf;
using testing::Pair;
using testing::UnorderedElementsAre;
using testing::UnorderedElementsAreArray;

namespace clang {
namespace clangd {
namespace {

const char *YAML = R"(
---
!Symbol
ID: 057557CEBF6E6B2D
Name:   'Foo1'
Scope:   'clang::'
SymInfo:
  Kind:            Function
  Lang:            Cpp
CanonicalDeclaration:
  FileURI:        file:///path/foo.h
  Start:
    Line: 1
    Column: 0
  End:
    Line: 1
    Column: 1
Origin:    128
Flags:    129
Documentation:    'Foo doc'
ReturnType:    'int'
IncludeHeaders:
  - Header:    'include1'
    References:    7
  - Header:    'include2'
    References:    3
...
---
!Symbol
ID: 057557CEBF6E6B2E
Name:   'Foo2'
Scope:   'clang::'
SymInfo:
  Kind:            Function
  Lang:            Cpp
CanonicalDeclaration:
  FileURI:        file:///path/bar.h
  Start:
    Line: 1
    Column: 0
  End:
    Line: 1
    Column: 1
Flags:    2
Signature:    '-sig'
CompletionSnippetSuffix:    '-snippet'
...
!Refs
ID: 057557CEBF6E6B2D
References:
  - Kind: 4
    Location:
      FileURI:    file:///path/foo.cc
      Start:
        Line: 5
        Column: 3
      End:
        Line: 5
        Column: 8
)";

MATCHER_P(ID, I, "") { return arg.ID == cantFail(SymbolID::fromStr(I)); }
MATCHER_P(QName, Name, "") { return (arg.Scope + arg.Name).str() == Name; }
MATCHER_P2(IncludeHeaderWithRef, IncludeHeader, References, "") {
  return (arg.IncludeHeader == IncludeHeader) && (arg.References == References);
}

TEST(SerializationTest, NoCrashOnEmptyYAML) {
  EXPECT_TRUE(bool(readIndexFile("")));
}

TEST(SerializationTest, YAMLConversions) {
  auto In = readIndexFile(YAML);
  EXPECT_TRUE(bool(In)) << In.takeError();

  auto ParsedYAML = readIndexFile(YAML);
  ASSERT_TRUE(bool(ParsedYAML)) << ParsedYAML.takeError();
  ASSERT_TRUE(bool(ParsedYAML->Symbols));
  EXPECT_THAT(
      *ParsedYAML->Symbols,
      UnorderedElementsAre(ID("057557CEBF6E6B2D"), ID("057557CEBF6E6B2E")));

  auto Sym1 = *ParsedYAML->Symbols->find(
      cantFail(SymbolID::fromStr("057557CEBF6E6B2D")));
  auto Sym2 = *ParsedYAML->Symbols->find(
      cantFail(SymbolID::fromStr("057557CEBF6E6B2E")));

  EXPECT_THAT(Sym1, QName("clang::Foo1"));
  EXPECT_EQ(Sym1.Signature, "");
  EXPECT_EQ(Sym1.Documentation, "Foo doc");
  EXPECT_EQ(Sym1.ReturnType, "int");
  EXPECT_EQ(StringRef(Sym1.CanonicalDeclaration.FileURI), "file:///path/foo.h");
  EXPECT_EQ(Sym1.Origin, static_cast<SymbolOrigin>(1 << 7));
  EXPECT_EQ(static_cast<uint8_t>(Sym1.Flags), 129);
  EXPECT_TRUE(Sym1.Flags & Symbol::IndexedForCodeCompletion);
  EXPECT_FALSE(Sym1.Flags & Symbol::Deprecated);
  EXPECT_THAT(Sym1.IncludeHeaders,
              UnorderedElementsAre(IncludeHeaderWithRef("include1", 7u),
                                   IncludeHeaderWithRef("include2", 3u)));

  EXPECT_THAT(Sym2, QName("clang::Foo2"));
  EXPECT_EQ(Sym2.Signature, "-sig");
  EXPECT_EQ(Sym2.ReturnType, "");
  EXPECT_EQ(llvm::StringRef(Sym2.CanonicalDeclaration.FileURI),
            "file:///path/bar.h");
  EXPECT_FALSE(Sym2.Flags & Symbol::IndexedForCodeCompletion);
  EXPECT_TRUE(Sym2.Flags & Symbol::Deprecated);

  ASSERT_TRUE(bool(ParsedYAML->Refs));
  EXPECT_THAT(
      *ParsedYAML->Refs,
      UnorderedElementsAre(Pair(cantFail(SymbolID::fromStr("057557CEBF6E6B2D")),
                                testing::SizeIs(1))));
  auto Ref1 = ParsedYAML->Refs->begin()->second.front();
  EXPECT_EQ(Ref1.Kind, RefKind::Reference);
  EXPECT_EQ(StringRef(Ref1.Location.FileURI), "file:///path/foo.cc");
}

std::vector<std::string> YAMLFromSymbols(const SymbolSlab &Slab) {
  std::vector<std::string> Result;
  for (const auto &Sym : Slab)
    Result.push_back(toYAML(Sym));
  return Result;
}
std::vector<std::string> YAMLFromRefs(const RefSlab &Slab) {
  std::vector<std::string> Result;
  for (const auto &Sym : Slab)
    Result.push_back(toYAML(Sym));
  return Result;
}

TEST(SerializationTest, BinaryConversions) {
  auto In = readIndexFile(YAML);
  EXPECT_TRUE(bool(In)) << In.takeError();

  // Write to binary format, and parse again.
  IndexFileOut Out(*In);
  Out.Format = IndexFileFormat::RIFF;
  std::string Serialized = llvm::to_string(Out);

  auto In2 = readIndexFile(Serialized);
  ASSERT_TRUE(bool(In2)) << In.takeError();
  ASSERT_TRUE(In2->Symbols);
  ASSERT_TRUE(In2->Refs);

  // Assert the YAML serializations match, for nice comparisons and diffs.
  EXPECT_THAT(YAMLFromSymbols(*In2->Symbols),
              UnorderedElementsAreArray(YAMLFromSymbols(*In->Symbols)));
  EXPECT_THAT(YAMLFromRefs(*In2->Refs),
              UnorderedElementsAreArray(YAMLFromRefs(*In->Refs)));
}

TEST(SerializationTest, SrcsTest) {
  auto In = readIndexFile(YAML);
  EXPECT_TRUE(bool(In)) << In.takeError();

  std::string TestContent("TestContent");
  IncludeGraphNode IGN;
  IGN.Digest =
      llvm::SHA1::hash({reinterpret_cast<const uint8_t *>(TestContent.data()),
                        TestContent.size()});
  IGN.DirectIncludes = {"inc1", "inc2"};
  IGN.URI = "URI";
  IGN.IsTU = true;
  IncludeGraph Sources;
  Sources[IGN.URI] = IGN;
  // Write to binary format, and parse again.
  IndexFileOut Out(*In);
  Out.Format = IndexFileFormat::RIFF;
  Out.Sources = &Sources;
  {
    std::string Serialized = llvm::to_string(Out);

    auto In = readIndexFile(Serialized);
    ASSERT_TRUE(bool(In)) << In.takeError();
    ASSERT_TRUE(In->Symbols);
    ASSERT_TRUE(In->Refs);
    ASSERT_TRUE(In->Sources);
    ASSERT_TRUE(In->Sources->count(IGN.URI));
    // Assert the YAML serializations match, for nice comparisons and diffs.
    EXPECT_THAT(YAMLFromSymbols(*In->Symbols),
                UnorderedElementsAreArray(YAMLFromSymbols(*In->Symbols)));
    EXPECT_THAT(YAMLFromRefs(*In->Refs),
                UnorderedElementsAreArray(YAMLFromRefs(*In->Refs)));
    auto IGNDeserialized = In->Sources->lookup(IGN.URI);
    EXPECT_EQ(IGNDeserialized.Digest, IGN.Digest);
    EXPECT_EQ(IGNDeserialized.DirectIncludes, IGN.DirectIncludes);
    EXPECT_EQ(IGNDeserialized.URI, IGN.URI);
    EXPECT_EQ(IGNDeserialized.IsTU, IGN.IsTU);
  }
}

} // namespace
} // namespace clangd
} // namespace clang
