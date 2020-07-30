//===--- MarshallingTests.cpp ------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../TestTU.h"
#include "TestFS.h"
#include "index/Index.h"
#include "index/Ref.h"
#include "index/Serialization.h"
#include "index/Symbol.h"
#include "index/SymbolID.h"
#include "index/remote/marshalling/Marshalling.h"
#include "clang/Index/IndexSymbol.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/StringSaver.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cstring>

namespace clang {
namespace clangd {
namespace remote {
namespace {

using llvm::sys::path::convert_to_slash;

const char *testPathURI(llvm::StringRef Path,
                        llvm::UniqueStringSaver &Strings) {
  auto URI = URI::createFile(testPath(Path));
  return Strings.save(URI.toString()).begin();
}

TEST(RemoteMarshallingTest, URITranslation) {
  llvm::BumpPtrAllocator Arena;
  llvm::UniqueStringSaver Strings(Arena);
  clangd::Ref Original;
  Original.Location.FileURI =
      testPathURI("remote/machine/projects/llvm-project/clang-tools-extra/"
                  "clangd/unittests/remote/MarshallingTests.cpp",
                  Strings);
  auto Serialized =
      toProtobuf(Original, testPath("remote/machine/projects/llvm-project/"));
  EXPECT_EQ(Serialized.location().file_path(),
            "clang-tools-extra/clangd/unittests/remote/MarshallingTests.cpp");
  const std::string LocalIndexPrefix = testPath("local/machine/project/");
  auto Deserialized = fromProtobuf(Serialized, &Strings,
                                   testPath("home/my-projects/llvm-project/"));
  EXPECT_TRUE(Deserialized);
  EXPECT_EQ(Deserialized->Location.FileURI,
            testPathURI("home/my-projects/llvm-project/clang-tools-extra/"
                        "clangd/unittests/remote/MarshallingTests.cpp",
                        Strings));

  clangd::Ref WithInvalidURI;
  // Invalid URI results in empty path.
  WithInvalidURI.Location.FileURI = "This is not a URI";
  Serialized = toProtobuf(WithInvalidURI, testPath("home/"));
  EXPECT_EQ(Serialized.location().file_path(), "");

  // Can not use URIs with scheme different from "file".
  auto UnittestURI =
      URI::create(testPath("project/lib/HelloWorld.cpp"), "unittest");
  EXPECT_TRUE(bool(UnittestURI));
  WithInvalidURI.Location.FileURI =
      Strings.save(UnittestURI->toString()).begin();
  Serialized = toProtobuf(WithInvalidURI, testPath("project/lib/"));
  EXPECT_EQ(Serialized.location().file_path(), "");

  Ref WithAbsolutePath;
  *WithAbsolutePath.mutable_location()->mutable_file_path() =
      "/usr/local/user/home/HelloWorld.cpp";
  Deserialized = fromProtobuf(WithAbsolutePath, &Strings, LocalIndexPrefix);
  // Paths transmitted over the wire can not be absolute, they have to be
  // relative.
  EXPECT_FALSE(Deserialized);
}

TEST(RemoteMarshallingTest, SymbolSerialization) {
  clangd::Symbol Sym;

  auto ID = SymbolID::fromStr("057557CEBF6E6B2D");
  EXPECT_TRUE(bool(ID));
  Sym.ID = *ID;

  index::SymbolInfo Info;
  Info.Kind = index::SymbolKind::Function;
  Info.SubKind = index::SymbolSubKind::AccessorGetter;
  Info.Lang = index::SymbolLanguage::CXX;
  Info.Properties = static_cast<index::SymbolPropertySet>(
      index::SymbolProperty::TemplateSpecialization);
  Sym.SymInfo = Info;

  llvm::BumpPtrAllocator Arena;
  llvm::UniqueStringSaver Strings(Arena);

  Sym.Name = Strings.save("Foo");
  Sym.Scope = Strings.save("llvm::foo::bar::");

  clangd::SymbolLocation Location;
  Location.Start.setLine(1);
  Location.Start.setColumn(15);
  Location.End.setLine(3);
  Location.End.setColumn(121);
  Location.FileURI = testPathURI("home/Definition.cpp", Strings);
  Sym.Definition = Location;

  Location.Start.setLine(42);
  Location.Start.setColumn(31);
  Location.End.setLine(20);
  Location.End.setColumn(400);
  Location.FileURI = testPathURI("home/Declaration.h", Strings);
  Sym.CanonicalDeclaration = Location;

  Sym.References = 9000;
  Sym.Origin = clangd::SymbolOrigin::Static;
  Sym.Signature = Strings.save("(int X, char Y, Type T)");
  Sym.TemplateSpecializationArgs = Strings.save("<int, char, bool, Type>");
  Sym.CompletionSnippetSuffix =
      Strings.save("({1: int X}, {2: char Y}, {3: Type T})");
  Sym.Documentation = Strings.save("This is my amazing Foo constructor!");
  Sym.ReturnType = Strings.save("Foo");

  Sym.Flags = clangd::Symbol::SymbolFlag::IndexedForCodeCompletion;

  // Check that symbols are exactly the same if the path to indexed project is
  // the same on indexing machine and the client.
  auto Serialized = toProtobuf(Sym, testPath("home/"));
  auto Deserialized = fromProtobuf(Serialized, &Strings, testPath("home/"));
  EXPECT_TRUE(Deserialized);
  EXPECT_EQ(toYAML(Sym), toYAML(*Deserialized));
  // Serialized paths are relative and have UNIX slashes.
  EXPECT_EQ(convert_to_slash(Serialized.definition().file_path(),
                             llvm::sys::path::Style::posix),
            Serialized.definition().file_path());
  EXPECT_TRUE(
      llvm::sys::path::is_relative(Serialized.definition().file_path()));

  // Fail with an invalid URI.
  Location.FileURI = "Not A URI";
  Sym.Definition = Location;
  Serialized = toProtobuf(Sym, testPath("home/"));
  Deserialized = fromProtobuf(Serialized, &Strings, testPath("home/"));
  EXPECT_FALSE(Deserialized);

  // Schemes other than "file" can not be used.
  auto UnittestURI = URI::create(testPath("home/SomePath.h"), "unittest");
  EXPECT_TRUE(bool(UnittestURI));
  Location.FileURI = Strings.save(UnittestURI->toString()).begin();
  Sym.Definition = Location;
  Serialized = toProtobuf(Sym, testPath("home/"));
  Deserialized = fromProtobuf(Serialized, &Strings, testPath("home/"));
  EXPECT_FALSE(Deserialized);

  // Passing root that is not prefix of the original file path.
  Location.FileURI = testPathURI("home/File.h", Strings);
  Sym.Definition = Location;
  // Check that the symbol is valid and passing the correct path works.
  Serialized = toProtobuf(Sym, testPath("home/"));
  Deserialized = fromProtobuf(Serialized, &Strings, testPath("home/"));
  EXPECT_TRUE(Deserialized);
  EXPECT_EQ(Deserialized->Definition.FileURI,
            testPathURI("home/File.h", Strings));
  // Fail with a wrong root.
  Serialized = toProtobuf(Sym, testPath("nothome/"));
  Deserialized = fromProtobuf(Serialized, &Strings, testPath("home/"));
  EXPECT_FALSE(Deserialized);
}

TEST(RemoteMarshallingTest, RefSerialization) {
  clangd::Ref Ref;
  Ref.Kind = clangd::RefKind::Spelled | clangd::RefKind::Declaration;

  llvm::BumpPtrAllocator Arena;
  llvm::UniqueStringSaver Strings(Arena);

  clangd::SymbolLocation Location;
  Location.Start.setLine(124);
  Location.Start.setColumn(21);
  Location.End.setLine(3213);
  Location.End.setColumn(541);
  Location.FileURI = testPathURI(
      "llvm-project/llvm/clang-tools-extra/clangd/Protocol.h", Strings);
  Ref.Location = Location;

  auto Serialized = toProtobuf(Ref, testPath("llvm-project/"));
  auto Deserialized =
      fromProtobuf(Serialized, &Strings, testPath("llvm-project/"));
  EXPECT_TRUE(Deserialized);
  EXPECT_EQ(toYAML(Ref), toYAML(*Deserialized));
}

TEST(RemoteMarshallingTest, IncludeHeaderURIs) {
  llvm::BumpPtrAllocator Arena;
  llvm::UniqueStringSaver Strings(Arena);

  llvm::SmallVector<clangd::Symbol::IncludeHeaderWithReferences, 1>
      ValidHeaders;
  clangd::Symbol::IncludeHeaderWithReferences Header;
  Header.IncludeHeader = Strings.save(
      URI::createFile("/usr/local/user/home/project/Header.h").toString());
  Header.References = 21;
  ValidHeaders.push_back(Header);
  Header.IncludeHeader = Strings.save("<iostream>");
  Header.References = 100;
  ValidHeaders.push_back(Header);
  Header.IncludeHeader = Strings.save("\"cstdio\"");
  Header.References = 200;
  ValidHeaders.push_back(Header);

  llvm::SmallVector<clangd::Symbol::IncludeHeaderWithReferences, 1>
      InvalidHeaders;
  // This is an absolute path to a header: can not be transmitted over the wire.
  Header.IncludeHeader = Strings.save(testPath("project/include/Common.h"));
  Header.References = 42;
  InvalidHeaders.push_back(Header);
  // This is not a valid header: can not be transmitted over the wire;
  Header.IncludeHeader = Strings.save("NotAHeader");
  Header.References = 5;
  InvalidHeaders.push_back(Header);

  clangd::Symbol Sym;
  // Fill in definition and declaration, Symbool will be invalid otherwise.
  clangd::SymbolLocation Location;
  Location.Start.setLine(1);
  Location.Start.setColumn(2);
  Location.End.setLine(3);
  Location.End.setColumn(4);
  Location.FileURI = testPathURI("File.h", Strings);
  Sym.Definition = Location;
  Sym.CanonicalDeclaration = Location;

  // Try to serialize all headers but only valid ones will end up in Protobuf
  // message.
  auto AllHeaders = ValidHeaders;
  AllHeaders.insert(AllHeaders.end(), InvalidHeaders.begin(),
                    InvalidHeaders.end());
  Sym.IncludeHeaders = AllHeaders;

  auto Serialized = toProtobuf(Sym, convert_to_slash("/"));
  EXPECT_EQ(static_cast<size_t>(Serialized.headers_size()),
            ValidHeaders.size());
  auto Deserialized = fromProtobuf(Serialized, &Strings, convert_to_slash("/"));
  EXPECT_TRUE(Deserialized);

  Sym.IncludeHeaders = ValidHeaders;
  EXPECT_EQ(toYAML(Sym), toYAML(*Deserialized));
}

TEST(RemoteMarshallingTest, FuzzyFindRequestSerialization) {
  clangd::FuzzyFindRequest Request;
  Request.ProximityPaths = {testPath("remote/Header.h"),
                            testPath("remote/subdir/OtherHeader.h"),
                            testPath("notremote/File.h"), "Not a Path."};
  auto Serialized = toProtobuf(Request, testPath("remote/"));
  EXPECT_EQ(Serialized.proximity_paths_size(), 2);
  auto Deserialized = fromProtobuf(&Serialized, testPath("home/"));
  EXPECT_THAT(Deserialized.ProximityPaths,
              testing::ElementsAre(testPath("home/Header.h"),
                                   testPath("home/subdir/OtherHeader.h")));
}

TEST(RemoteMarshallingTest, RelativePathToURITranslation) {
  EXPECT_TRUE(relativePathToURI("lib/File.cpp", testPath("home/project/")));
  // RelativePath can not be absolute.
  EXPECT_FALSE(relativePathToURI("/lib/File.cpp", testPath("home/project/")));
  // IndexRoot has to be absolute path.
  EXPECT_FALSE(relativePathToURI("lib/File.cpp", "home/project/"));
}

TEST(RemoteMarshallingTest, URIToRelativePathTranslation) {
  llvm::BumpPtrAllocator Arena;
  llvm::UniqueStringSaver Strings(Arena);
  EXPECT_TRUE(
      uriToRelativePath(testPathURI("home/project/lib/File.cpp", Strings),
                        testPath("home/project/")));
  // IndexRoot has to be absolute path.
  EXPECT_FALSE(uriToRelativePath(
      testPathURI("home/project/lib/File.cpp", Strings), "home/project/"));
  // IndexRoot has to be be a prefix of the file path.
  EXPECT_FALSE(
      uriToRelativePath(testPathURI("home/project/lib/File.cpp", Strings),
                        testPath("home/other/project/")));
}

} // namespace
} // namespace remote
} // namespace clangd
} // namespace clang
