//===-- SerializationTests.cpp - Binary and YAML serialization unit tests -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "index/Serialization.h"
#include "index/SymbolYAML.h"
#include "llvm/Support/ScopedPrinter.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using testing::UnorderedElementsAre;
using testing::UnorderedElementsAreArray;
namespace clang {
namespace clangd {
namespace {

const char *YAML1 = R"(
---
ID: 057557CEBF6E6B2DD437FBF60CC58F352D1DF856
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
Flags:    1
Documentation:    'Foo doc'
ReturnType:    'int'
IncludeHeaders:
  - Header:    'include1'
    References:    7
  - Header:    'include2'
    References:    3
...
)";

const char *YAML2 = R"(
---
ID: 057557CEBF6E6B2DD437FBF60CC58F352D1DF858
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
)";

MATCHER_P(QName, Name, "") { return (arg.Scope + arg.Name).str() == Name; }
MATCHER_P2(IncludeHeaderWithRef, IncludeHeader, References, "") {
  return (arg.IncludeHeader == IncludeHeader) && (arg.References == References);
}

TEST(SerializationTest, YAMLConversions) {
  auto Symbols1 = symbolsFromYAML(YAML1);
  ASSERT_EQ(Symbols1.size(), 1u);
  const auto &Sym1 = *Symbols1.begin();
  EXPECT_THAT(Sym1, QName("clang::Foo1"));
  EXPECT_EQ(Sym1.Signature, "");
  EXPECT_EQ(Sym1.Documentation, "Foo doc");
  EXPECT_EQ(Sym1.ReturnType, "int");
  EXPECT_EQ(Sym1.CanonicalDeclaration.FileURI, "file:///path/foo.h");
  EXPECT_TRUE(Sym1.Flags & Symbol::IndexedForCodeCompletion);
  EXPECT_FALSE(Sym1.Flags & Symbol::Deprecated);
  EXPECT_THAT(Sym1.IncludeHeaders,
              UnorderedElementsAre(IncludeHeaderWithRef("include1", 7u),
                                   IncludeHeaderWithRef("include2", 3u)));

  auto Symbols2 = symbolsFromYAML(YAML2);
  ASSERT_EQ(Symbols2.size(), 1u);
  const auto &Sym2 = *Symbols2.begin();
  EXPECT_THAT(Sym2, QName("clang::Foo2"));
  EXPECT_EQ(Sym2.Signature, "-sig");
  EXPECT_EQ(Sym2.ReturnType, "");
  EXPECT_EQ(Sym2.CanonicalDeclaration.FileURI, "file:///path/bar.h");
  EXPECT_FALSE(Sym2.Flags & Symbol::IndexedForCodeCompletion);
  EXPECT_TRUE(Sym2.Flags & Symbol::Deprecated);

  std::string ConcatenatedYAML;
  {
    llvm::raw_string_ostream OS(ConcatenatedYAML);
    SymbolsToYAML(Symbols1, OS);
    SymbolsToYAML(Symbols2, OS);
  }
  auto ConcatenatedSymbols = symbolsFromYAML(ConcatenatedYAML);
  EXPECT_THAT(ConcatenatedSymbols,
              UnorderedElementsAre(QName("clang::Foo1"), QName("clang::Foo2")));
}

std::vector<std::string> YAMLFromSymbols(const SymbolSlab &Slab) {
  std::vector<std::string> Result;
  for (const auto &Sym : Slab)
    Result.push_back(SymbolToYAML(Sym));
  return Result;
}

TEST(SerializationTest, BinaryConversions) {
  // We reuse the test symbols from YAML.
  auto Slab = symbolsFromYAML(std::string(YAML1) + YAML2);
  ASSERT_EQ(Slab.size(), 2u);

  // Write to binary format, and parse again.
  IndexFileOut Out;
  Out.Symbols = &Slab;
  std::string Serialized = llvm::to_string(Out);

  auto In = readIndexFile(Serialized);
  ASSERT_TRUE(bool(In)) << In.takeError();
  ASSERT_TRUE(In->Symbols);

  // Assert the YAML serializations match, for nice comparisons and diffs.
  EXPECT_THAT(YAMLFromSymbols(*In->Symbols),
              UnorderedElementsAreArray(YAMLFromSymbols(Slab)));
}

} // namespace
} // namespace clangd
} // namespace clang
