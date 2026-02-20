//===-- SerializationTests.cpp - Binary and YAML serialization unit tests -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Headers.h"
#include "RIFF.h"
#include "index/Serialization.h"
#include "support/Logger.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Config/llvm-config.h" // for LLVM_ON_UNIX
#include "llvm/Support/Compression.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ScopedPrinter.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#ifdef LLVM_ON_UNIX
#include <sys/resource.h>
#endif

using ::testing::ElementsAre;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;
using ::testing::UnorderedElementsAreArray;

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
Flags:    129
Documentation:    'Foo doc'
ReturnType:    'int'
IncludeHeaders:
  - Header:    'include1'
    References:    7
    Directives:      [ Include ]
  - Header:    'include2'
    References:    3
    Directives:      [ Import ]
  - Header:    'include3'
    References:    2
    Directives:      [ Include, Import ]
  - Header:    'include4'
    References:    1
    Directives:      [ ]
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
...
--- !Relations
Subject:
  ID:              6481EE7AF2841756
Predicate:       0
Object:
  ID:              6512AEC512EA3A2D
...
--- !Cmd
Directory:       'testdir'
CommandLine:
  - 'cmd1'
  - 'cmd2'
...
--- !Source
URI:             'file:///path/source1.cpp'
Flags:           1
Digest:          EED8F5EAF25C453C
DirectIncludes:
  - 'file:///path/inc1.h'
  - 'file:///path/inc2.h'
...
)";

MATCHER_P(id, I, "") { return arg.ID == cantFail(SymbolID::fromStr(I)); }
MATCHER_P(qName, Name, "") { return (arg.Scope + arg.Name).str() == Name; }
MATCHER_P3(IncludeHeaderWithRefAndDirectives, IncludeHeader, References,
           SupportedDirectives, "") {
  return (arg.IncludeHeader == IncludeHeader) &&
         (arg.References == References) &&
         (arg.SupportedDirectives == SupportedDirectives);
}

auto readIndexFile(llvm::StringRef Text) {
  return readIndexFile(Text, SymbolOrigin::Static);
}

TEST(SerializationTest, NoCrashOnEmptyYAML) {
  EXPECT_TRUE(bool(readIndexFile("")));
}

TEST(SerializationTest, YAMLConversions) {
  auto ParsedYAML = readIndexFile(YAML);
  ASSERT_TRUE(bool(ParsedYAML)) << ParsedYAML.takeError();
  ASSERT_TRUE(bool(ParsedYAML->Symbols));
  EXPECT_THAT(
      *ParsedYAML->Symbols,
      UnorderedElementsAre(id("057557CEBF6E6B2D"), id("057557CEBF6E6B2E")));

  auto Sym1 = *ParsedYAML->Symbols->find(
      cantFail(SymbolID::fromStr("057557CEBF6E6B2D")));
  auto Sym2 = *ParsedYAML->Symbols->find(
      cantFail(SymbolID::fromStr("057557CEBF6E6B2E")));

  EXPECT_THAT(Sym1, qName("clang::Foo1"));
  EXPECT_EQ(Sym1.Signature, "");
  EXPECT_EQ(Sym1.Documentation, "Foo doc");
  EXPECT_EQ(Sym1.ReturnType, "int");
  EXPECT_EQ(StringRef(Sym1.CanonicalDeclaration.FileURI), "file:///path/foo.h");
  EXPECT_EQ(Sym1.Origin, SymbolOrigin::Static);
  EXPECT_EQ(static_cast<uint8_t>(Sym1.Flags), 129);
  EXPECT_TRUE(Sym1.Flags & Symbol::IndexedForCodeCompletion);
  EXPECT_FALSE(Sym1.Flags & Symbol::Deprecated);
  EXPECT_THAT(
      Sym1.IncludeHeaders,
      UnorderedElementsAre(
          IncludeHeaderWithRefAndDirectives("include1", 7u, Symbol::Include),
          IncludeHeaderWithRefAndDirectives("include2", 3u, Symbol::Import),
          IncludeHeaderWithRefAndDirectives("include3", 2u,
                                            Symbol::Include | Symbol::Import),
          IncludeHeaderWithRefAndDirectives("include4", 1u, Symbol::Invalid)));

  EXPECT_THAT(Sym2, qName("clang::Foo2"));
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
                                ::testing::SizeIs(1))));
  auto Ref1 = ParsedYAML->Refs->begin()->second.front();
  EXPECT_EQ(Ref1.Kind, RefKind::Reference);
  EXPECT_EQ(StringRef(Ref1.Location.FileURI), "file:///path/foo.cc");

  SymbolID Base = cantFail(SymbolID::fromStr("6481EE7AF2841756"));
  SymbolID Derived = cantFail(SymbolID::fromStr("6512AEC512EA3A2D"));
  ASSERT_TRUE(bool(ParsedYAML->Relations));
  EXPECT_THAT(
      *ParsedYAML->Relations,
      UnorderedElementsAre(Relation{Base, RelationKind::BaseOf, Derived}));

  ASSERT_TRUE(bool(ParsedYAML->Cmd));
  auto &Cmd = *ParsedYAML->Cmd;
  ASSERT_EQ(Cmd.Directory, "testdir");
  EXPECT_THAT(Cmd.CommandLine, ElementsAre("cmd1", "cmd2"));

  ASSERT_TRUE(bool(ParsedYAML->Sources));
  const auto *URI = "file:///path/source1.cpp";
  ASSERT_TRUE(ParsedYAML->Sources->count(URI));
  auto IGNDeserialized = ParsedYAML->Sources->lookup(URI);
  EXPECT_EQ(llvm::toHex(IGNDeserialized.Digest), "EED8F5EAF25C453C");
  EXPECT_THAT(IGNDeserialized.DirectIncludes,
              ElementsAre("file:///path/inc1.h", "file:///path/inc2.h"));
  EXPECT_EQ(IGNDeserialized.URI, URI);
  EXPECT_EQ(IGNDeserialized.Flags, IncludeGraphNode::SourceFlag(1));
}

std::vector<std::string> yamlFromSymbols(const SymbolSlab &Slab) {
  std::vector<std::string> Result;
  for (const auto &Sym : Slab)
    Result.push_back(toYAML(Sym));
  return Result;
}
std::vector<std::string> yamlFromRefs(const RefSlab &Slab) {
  std::vector<std::string> Result;
  for (const auto &Refs : Slab)
    Result.push_back(toYAML(Refs));
  return Result;
}

std::vector<std::string> yamlFromRelations(const RelationSlab &Slab) {
  std::vector<std::string> Result;
  for (const auto &Rel : Slab)
    Result.push_back(toYAML(Rel));
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
  ASSERT_TRUE(bool(In2)) << In2.takeError();
  ASSERT_TRUE(In2->Symbols);
  ASSERT_TRUE(In2->Refs);
  ASSERT_TRUE(In2->Relations);

  // Assert the YAML serializations match, for nice comparisons and diffs.
  EXPECT_THAT(yamlFromSymbols(*In2->Symbols),
              UnorderedElementsAreArray(yamlFromSymbols(*In->Symbols)));
  EXPECT_THAT(yamlFromRefs(*In2->Refs),
              UnorderedElementsAreArray(yamlFromRefs(*In->Refs)));
  EXPECT_THAT(yamlFromRelations(*In2->Relations),
              UnorderedElementsAreArray(yamlFromRelations(*In->Relations)));
}

TEST(SerializationTest, SrcsTest) {
  auto In = readIndexFile(YAML);
  EXPECT_TRUE(bool(In)) << In.takeError();

  std::string TestContent("TestContent");
  IncludeGraphNode IGN;
  IGN.Digest = digest(TestContent);
  IGN.DirectIncludes = {"inc1", "inc2"};
  IGN.URI = "URI";
  IGN.Flags |= IncludeGraphNode::SourceFlag::IsTU;
  IGN.Flags |= IncludeGraphNode::SourceFlag::HadErrors;
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
    EXPECT_THAT(yamlFromSymbols(*In->Symbols),
                UnorderedElementsAreArray(yamlFromSymbols(*In->Symbols)));
    EXPECT_THAT(yamlFromRefs(*In->Refs),
                UnorderedElementsAreArray(yamlFromRefs(*In->Refs)));
    auto IGNDeserialized = In->Sources->lookup(IGN.URI);
    EXPECT_EQ(IGNDeserialized.Digest, IGN.Digest);
    EXPECT_EQ(IGNDeserialized.DirectIncludes, IGN.DirectIncludes);
    EXPECT_EQ(IGNDeserialized.URI, IGN.URI);
    EXPECT_EQ(IGNDeserialized.Flags, IGN.Flags);
  }
}

TEST(SerializationTest, CmdlTest) {
  auto In = readIndexFile(YAML);
  EXPECT_TRUE(bool(In)) << In.takeError();

  tooling::CompileCommand Cmd;
  Cmd.Directory = "testdir";
  Cmd.CommandLine.push_back("cmd1");
  Cmd.CommandLine.push_back("cmd2");
  Cmd.Filename = "ignored";
  Cmd.Heuristic = "ignored";
  Cmd.Output = "ignored";

  IndexFileOut Out(*In);
  Out.Format = IndexFileFormat::RIFF;
  Out.Cmd = &Cmd;
  {
    std::string Serialized = llvm::to_string(Out);

    auto In = readIndexFile(Serialized);
    ASSERT_TRUE(bool(In)) << In.takeError();
    ASSERT_TRUE(In->Cmd);

    const tooling::CompileCommand &SerializedCmd = *In->Cmd;
    EXPECT_EQ(SerializedCmd.CommandLine, Cmd.CommandLine);
    EXPECT_EQ(SerializedCmd.Directory, Cmd.Directory);
    EXPECT_NE(SerializedCmd.Filename, Cmd.Filename);
    EXPECT_NE(SerializedCmd.Heuristic, Cmd.Heuristic);
    EXPECT_NE(SerializedCmd.Output, Cmd.Output);
  }
}

// rlimit is part of POSIX. RLIMIT_AS does not exist in OpenBSD.
// Sanitizers use a lot of address space, so we can't apply strict limits.
#if LLVM_ON_UNIX && defined(RLIMIT_AS) && !LLVM_ADDRESS_SANITIZER_BUILD &&     \
    !LLVM_MEMORY_SANITIZER_BUILD && !LLVM_THREAD_SANITIZER_BUILD
class ScopedMemoryLimit {
  struct rlimit OriginalLimit;
  bool Succeeded = false;

public:
  ScopedMemoryLimit(rlim_t Bytes) {
    if (!getrlimit(RLIMIT_AS, &OriginalLimit)) {
      struct rlimit NewLimit = OriginalLimit;
      NewLimit.rlim_cur = Bytes;
      Succeeded = !setrlimit(RLIMIT_AS, &NewLimit);
    }
    if (!Succeeded)
      log("Failed to set rlimit");
  }

  ~ScopedMemoryLimit() {
    if (Succeeded)
      setrlimit(RLIMIT_AS, &OriginalLimit);
  }
};
#else
class ScopedMemoryLimit {
public:
  ScopedMemoryLimit(unsigned Bytes) { log("rlimit unsupported"); }
};
#endif

// Test that our deserialization detects invalid array sizes without allocating.
// If this detection fails, the test should allocate a huge array and crash.
TEST(SerializationTest, NoCrashOnBadArraySize) {
  // This test is tricky because we need to construct a subtly invalid file.
  // First, create a valid serialized file.
  auto In = readIndexFile(YAML);
  ASSERT_FALSE(!In) << In.takeError();
  IndexFileOut Out(*In);
  Out.Format = IndexFileFormat::RIFF;
  std::string Serialized = llvm::to_string(Out);

  // Low-level parse it again and find the `srcs` chunk we're going to corrupt.
  auto Parsed = riff::readFile(Serialized);
  ASSERT_FALSE(!Parsed) << Parsed.takeError();
  auto Srcs = llvm::find_if(Parsed->Chunks, [](riff::Chunk C) {
    return C.ID == riff::fourCC("srcs");
  });
  ASSERT_NE(Srcs, Parsed->Chunks.end());

  // Srcs consists of a sequence of IncludeGraphNodes. In our case, just one.
  // The node has:
  //  - 1 byte: flags (1)
  //  - varint(stringID): URI
  //  - 8 byte: file digest
  //  - varint: DirectIncludes.length
  //  - repeated varint(stringID): DirectIncludes
  // We want to set DirectIncludes.length to a huge number.
  // The offset isn't trivial to find, so we use the file digest.
  std::string FileDigest = llvm::fromHex("EED8F5EAF25C453C");
  unsigned Pos = Srcs->Data.find_first_of(FileDigest);
  ASSERT_NE(Pos, StringRef::npos) << "Couldn't locate file digest";
  Pos += FileDigest.size();

  // Varints are little-endian base-128 numbers, where the top-bit of each byte
  // indicates whether there are more. ffffffff0f -> 0xffffffff.
  std::string CorruptSrcs =
      (Srcs->Data.take_front(Pos) + llvm::fromHex("ffffffff0f") +
       "some_random_garbage")
          .str();
  Srcs->Data = CorruptSrcs;

  // Try to crash rather than hang on large allocation.
  ScopedMemoryLimit MemLimit(1000 * 1024 * 1024); // 1GB

  std::string CorruptFile = llvm::to_string(*Parsed);
  auto CorruptParsed = readIndexFile(CorruptFile);
  ASSERT_TRUE(!CorruptParsed);
  EXPECT_EQ(llvm::toString(CorruptParsed.takeError()),
            "malformed or truncated include uri");
}

// Check we detect invalid string table size size without allocating it first.
// If this detection fails, the test should allocate a huge array and crash.
TEST(SerializationTest, NoCrashOnBadStringTableSize) {
  if (!llvm::compression::zlib::isAvailable()) {
    log("skipping test, no zlib");
    return;
  }

  // First, create a valid serialized file.
  auto In = readIndexFile(YAML);
  ASSERT_FALSE(!In) << In.takeError();
  IndexFileOut Out(*In);
  Out.Format = IndexFileFormat::RIFF;
  std::string Serialized = llvm::to_string(Out);

  // Low-level parse it again, we're going to replace the `stri` chunk.
  auto Parsed = riff::readFile(Serialized);
  ASSERT_FALSE(!Parsed) << Parsed.takeError();
  auto Stri = llvm::find_if(Parsed->Chunks, [](riff::Chunk C) {
    return C.ID == riff::fourCC("stri");
  });
  ASSERT_NE(Stri, Parsed->Chunks.end());

  // stri consists of an 8 byte uncompressed-size, and then compressed data.
  // We'll claim our small amount of data expands to 4GB
  std::string CorruptStri =
      (llvm::fromHex("ffffffff") + Stri->Data.drop_front(4)).str();
  Stri->Data = CorruptStri;

  // Try to crash rather than hang on large allocation.
  ScopedMemoryLimit MemLimit(1000 * 1024 * 1024); // 1GB

  std::string CorruptFile = llvm::to_string(*Parsed);
  auto CorruptParsed = readIndexFile(CorruptFile);
  ASSERT_TRUE(!CorruptParsed);
  EXPECT_THAT(llvm::toString(CorruptParsed.takeError()),
              testing::HasSubstr("bytes is implausible"));
}

// Verify path remapping is applied to all URI fields during load/store.
// An index is generated at /home/project. A second client at /workarea/project
// loads and re-stores the shards. On-disk content always contains the
// /home/project paths so the index remains portable.
TEST(SerializationTest, PathTransformRoundTrip) {
  // Store transform: map /workarea/project -> /home/project so that
  // on-disk content stays in the canonical /home/project paths.
  PathTransform StoreTransform =
      [](llvm::StringRef URI) -> std::optional<std::string> {
    std::string S = URI.str();
    size_t Pos = S.find("/workarea/project/");
    if (Pos != std::string::npos) {
      S.replace(Pos, strlen("/workarea/project/"), "/home/project/");
      return S;
    }
    return std::nullopt;
  };
  // Load transform: map /home/project -> /workarea/project so that
  // in-memory paths match the local filesystem.
  PathTransform LoadTransform =
      [](llvm::StringRef URI) -> std::optional<std::string> {
    std::string S = URI.str();
    size_t Pos = S.find("/home/project/");
    if (Pos != std::string::npos) {
      S.replace(Pos, strlen("/home/project/"), "/workarea/project/");
      return S;
    }
    return std::nullopt;
  };

  // The index is generated with /home/project paths.
  Symbol Sym;
  Sym.ID = cantFail(SymbolID::fromStr("057557CEBF6E6B2D"));
  Sym.Name = "TestFunc";
  Sym.Scope = "ns::";
  Sym.Definition.FileURI = "file:///home/project/def.cpp";
  Sym.CanonicalDeclaration.FileURI = "file:///home/project/decl.h";
  Sym.IncludeHeaders.push_back(
      {/*IncludeHeader=*/"file:///home/project/header.h",
       /*References=*/1,
       /*SupportedDirectives=*/Symbol::Include});
  Sym.IncludeHeaders.push_back(
      {/*IncludeHeader=*/"<system_header>", // Literal, should not be modified
       /*References=*/1,
       /*SupportedDirectives=*/Symbol::Include});

  SymbolSlab::Builder SymbolBuilder;
  SymbolBuilder.insert(Sym);
  SymbolSlab Symbols = std::move(SymbolBuilder).build();

  Ref R;
  R.Location.FileURI = "file:///home/project/ref.cpp";
  R.Kind = RefKind::Reference;
  RefSlab::Builder RefBuilder;
  RefBuilder.insert(Sym.ID, R);
  RefSlab Refs = std::move(RefBuilder).build();

  IncludeGraph Sources;
  IncludeGraphNode IGN;
  IGN.URI = "file:///home/project/source.cpp";
  IGN.Flags = IncludeGraphNode::SourceFlag::IsTU;
  IGN.Digest = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'};
  IGN.DirectIncludes = {"file:///home/project/inc1.h",
                        "file:///home/project/inc2.h"};
  Sources[IGN.URI] = IGN;

  tooling::CompileCommand Cmd;
  Cmd.Directory = "/home/project/src";
  Cmd.CommandLine = {"/usr/bin/clang++",
                     "-I/home/project/include",
                     "-isystem/home/project/sysinclude",
                     "-isystem",
                     "/home/project/sysinclude2",
                     "-DFOO=bar",
                     "-DROOT=/home/project/src",
                     "/home/project/src/test.cpp"};

  // Serialize the index directly (no transform) to produce the canonical
  // on-disk format containing /home/project paths.
  IndexFileOut Out;
  Out.Symbols = &Symbols;
  Out.Refs = &Refs;
  Out.Sources = &Sources;
  Out.Cmd = &Cmd;
  Out.Format = IndexFileFormat::RIFF;
  std::string Serialized = llvm::to_string(Out);

  // Verify the on-disk shard contains /home/project paths.
  auto Raw = readIndexFile(Serialized, SymbolOrigin::Background);
  ASSERT_TRUE(bool(Raw)) << Raw.takeError();
  ASSERT_TRUE(Raw->Symbols);
  EXPECT_EQ(llvm::StringRef(Raw->Symbols->find(Sym.ID)->Definition.FileURI),
            "file:///home/project/def.cpp")
      << "On-disk shard should contain /home/project paths";
  ASSERT_TRUE(Raw->Cmd);
  EXPECT_EQ(Raw->Cmd->Directory, "/home/project/src");

  // Load with the transform to map /home/project -> /workarea/project.
  auto In = readIndexFile(Serialized, SymbolOrigin::Background, &LoadTransform);
  ASSERT_TRUE(bool(In)) << In.takeError();

  ASSERT_TRUE(In->Symbols);
  auto &ReadSym = *In->Symbols->find(Sym.ID);
  EXPECT_EQ(llvm::StringRef(ReadSym.Definition.FileURI),
            "file:///workarea/project/def.cpp")
      << "Symbol.Definition.FileURI not mapped";
  EXPECT_EQ(llvm::StringRef(ReadSym.CanonicalDeclaration.FileURI),
            "file:///workarea/project/decl.h")
      << "Symbol.CanonicalDeclaration.FileURI not mapped";
  ASSERT_EQ(ReadSym.IncludeHeaders.size(), 2u);
  EXPECT_EQ(ReadSym.IncludeHeaders[0].IncludeHeader,
            "file:///workarea/project/header.h")
      << "Symbol.IncludeHeaders[0].IncludeHeader not mapped";
  EXPECT_EQ(ReadSym.IncludeHeaders[1].IncludeHeader, "<system_header>")
      << "Literal include header should not be modified";

  ASSERT_TRUE(In->Refs);
  ASSERT_EQ(In->Refs->numRefs(), 1u);
  auto RefIt = In->Refs->begin();
  EXPECT_EQ(RefIt->first, Sym.ID);
  ASSERT_EQ(RefIt->second.size(), 1u);
  EXPECT_EQ(llvm::StringRef(RefIt->second[0].Location.FileURI),
            "file:///workarea/project/ref.cpp")
      << "Ref.Location.FileURI not mapped";

  ASSERT_TRUE(In->Sources);
  auto SourceIt = In->Sources->find("file:///workarea/project/source.cpp");
  ASSERT_NE(SourceIt, In->Sources->end()) << "Source URI key not mapped";
  EXPECT_EQ(SourceIt->second.URI, "file:///workarea/project/source.cpp")
      << "IncludeGraphNode.URI not mapped";
  EXPECT_THAT(SourceIt->second.DirectIncludes,
              ElementsAre("file:///workarea/project/inc1.h",
                          "file:///workarea/project/inc2.h"))
      << "IncludeGraphNode.DirectIncludes not mapped";

  ASSERT_TRUE(In->Cmd);
  EXPECT_EQ(In->Cmd->Directory, "/workarea/project/src")
      << "Cmd.Directory not mapped";
  EXPECT_THAT(In->Cmd->CommandLine,
              ElementsAre("/usr/bin/clang++", "-I/workarea/project/include",
                          "-isystem/workarea/project/sysinclude", "-isystem",
                          "/workarea/project/sysinclude2", "-DFOO=bar",
                          "-DROOT=/workarea/project/src",
                          "/workarea/project/src/test.cpp"))
      << "Cmd.CommandLine not mapped";

  // Re-serialize with the store transform. On-disk content should be
  // back in /home/project paths so the index remains portable.
  IndexFileOut WorkareaOut;
  WorkareaOut.Symbols = &*In->Symbols;
  WorkareaOut.Refs = &*In->Refs;
  WorkareaOut.Sources = &*In->Sources;
  WorkareaOut.Cmd = In->Cmd ? &*In->Cmd : nullptr;
  WorkareaOut.Format = IndexFileFormat::RIFF;
  WorkareaOut.Transform = &StoreTransform;
  std::string WorkareaSerialized = llvm::to_string(WorkareaOut);

  auto Restored = readIndexFile(WorkareaSerialized, SymbolOrigin::Background);
  ASSERT_TRUE(bool(Restored)) << Restored.takeError();
  ASSERT_TRUE(Restored->Symbols);
  EXPECT_EQ(
      llvm::StringRef(Restored->Symbols->find(Sym.ID)->Definition.FileURI),
      "file:///home/project/def.cpp")
      << "Store transform should restore /home/project paths on disk";
  ASSERT_TRUE(Restored->Cmd);
  EXPECT_EQ(Restored->Cmd->Directory, "/home/project/src")
      << "Store transform should restore Cmd.Directory on disk";
  EXPECT_THAT(Restored->Cmd->CommandLine,
              ElementsAre("/usr/bin/clang++", "-I/home/project/include",
                          "-isystem/home/project/sysinclude", "-isystem",
                          "/home/project/sysinclude2", "-DFOO=bar",
                          "-DROOT=/home/project/src",
                          "/home/project/src/test.cpp"))
      << "Store transform should restore Cmd.CommandLine on disk";
}

} // namespace
} // namespace clangd
} // namespace clang
