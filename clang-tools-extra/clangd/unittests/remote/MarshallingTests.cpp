//===--- MarshallingTests.cpp ------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../TestTU.h"
#include "Index.pb.h"
#include "TestFS.h"
#include "index/Index.h"
#include "index/Ref.h"
#include "index/Relation.h"
#include "index/Serialization.h"
#include "index/Symbol.h"
#include "index/SymbolID.h"
#include "index/SymbolLocation.h"
#include "index/remote/marshalling/Marshalling.h"
#include "clang/Index/IndexSymbol.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
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

clangd::Symbol createSymbol(llvm::StringRef PathPrefix,
                            llvm::UniqueStringSaver &Strings) {
  clangd::Symbol Sym;
  Sym.ID = llvm::cantFail(SymbolID::fromStr("057557CEBF6E6B2D"));

  index::SymbolInfo Info;
  Info.Kind = index::SymbolKind::Function;
  Info.SubKind = index::SymbolSubKind::AccessorGetter;
  Info.Lang = index::SymbolLanguage::CXX;
  Info.Properties = static_cast<index::SymbolPropertySet>(
      index::SymbolProperty::TemplateSpecialization);
  Sym.SymInfo = Info;

  Sym.Name = Strings.save("Foo");
  Sym.Scope = Strings.save("llvm::foo::bar::");

  clangd::SymbolLocation Location;
  Location.Start.setLine(1);
  Location.Start.setColumn(15);
  Location.End.setLine(3);
  Location.End.setColumn(121);
  Location.FileURI = testPathURI(PathPrefix.str() + "Definition.cpp", Strings);
  Sym.Definition = Location;

  Location.Start.setLine(42);
  Location.Start.setColumn(31);
  Location.End.setLine(20);
  Location.End.setColumn(400);
  Location.FileURI = testPathURI(PathPrefix.str() + "Declaration.h", Strings);
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

  return Sym;
}

TEST(RemoteMarshallingTest, URITranslation) {
  llvm::BumpPtrAllocator Arena;
  llvm::UniqueStringSaver Strings(Arena);
  Marshaller ProtobufMarshaller(
      testPath("remote/machine/projects/llvm-project/"),
      testPath("home/my-projects/llvm-project/"));
  clangd::Ref Original;
  Original.Location.FileURI =
      testPathURI("remote/machine/projects/llvm-project/clang-tools-extra/"
                  "clangd/unittests/remote/MarshallingTests.cpp",
                  Strings);
  auto Serialized = ProtobufMarshaller.toProtobuf(Original);
  ASSERT_TRUE(bool(Serialized));
  EXPECT_EQ(Serialized->location().file_path(),
            "clang-tools-extra/clangd/unittests/remote/MarshallingTests.cpp");
  auto Deserialized = ProtobufMarshaller.fromProtobuf(*Serialized);
  ASSERT_TRUE(bool(Deserialized));
  EXPECT_STREQ(Deserialized->Location.FileURI,
               testPathURI("home/my-projects/llvm-project/clang-tools-extra/"
                           "clangd/unittests/remote/MarshallingTests.cpp",
                           Strings));

  // Can't have empty paths.
  *Serialized->mutable_location()->mutable_file_path() = std::string();
  Deserialized = ProtobufMarshaller.fromProtobuf(*Serialized);
  EXPECT_FALSE(bool(Deserialized));
  llvm::consumeError(Deserialized.takeError());

  clangd::Ref WithInvalidURI;
  // Invalid URI results in serialization failure.
  WithInvalidURI.Location.FileURI = "This is not a URI";
  auto DeserializedRef = ProtobufMarshaller.toProtobuf(WithInvalidURI);
  EXPECT_FALSE(bool(DeserializedRef));
  llvm::consumeError(DeserializedRef.takeError());

  // Can not use URIs with scheme different from "file".
  auto UnittestURI =
      URI::create(testPath("project/lib/HelloWorld.cpp"), "unittest");
  ASSERT_TRUE(bool(UnittestURI));
  WithInvalidURI.Location.FileURI =
      Strings.save(UnittestURI->toString()).begin();
  auto DeserializedSymbol = ProtobufMarshaller.toProtobuf(WithInvalidURI);
  EXPECT_FALSE(bool(DeserializedSymbol));
  llvm::consumeError(DeserializedSymbol.takeError());

  // Paths transmitted over the wire can not be absolute, they have to be
  // relative.
  Ref WithAbsolutePath;
  *WithAbsolutePath.mutable_location()->mutable_file_path() =
      "/usr/local/user/home/HelloWorld.cpp";
  Deserialized = ProtobufMarshaller.fromProtobuf(WithAbsolutePath);
  EXPECT_FALSE(bool(Deserialized));
  llvm::consumeError(Deserialized.takeError());
}

TEST(RemoteMarshallingTest, SymbolSerialization) {
  llvm::BumpPtrAllocator Arena;
  llvm::UniqueStringSaver Strings(Arena);

  clangd::Symbol Sym = createSymbol("home/", Strings);
  Marshaller ProtobufMarshaller(testPath("home/"), testPath("home/"));

  // Check that symbols are exactly the same if the path to indexed project is
  // the same on indexing machine and the client.
  auto Serialized = ProtobufMarshaller.toProtobuf(Sym);
  ASSERT_TRUE(bool(Serialized));
  auto Deserialized = ProtobufMarshaller.fromProtobuf(*Serialized);
  ASSERT_TRUE(bool(Deserialized));
  EXPECT_EQ(toYAML(Sym), toYAML(*Deserialized));
  // Serialized paths are relative and have UNIX slashes.
  EXPECT_EQ(convert_to_slash(Serialized->definition().file_path(),
                             llvm::sys::path::Style::posix),
            Serialized->definition().file_path());
  EXPECT_TRUE(
      llvm::sys::path::is_relative(Serialized->definition().file_path()));

  // Missing definition is OK.
  Sym.Definition = clangd::SymbolLocation();
  Serialized = ProtobufMarshaller.toProtobuf(Sym);
  ASSERT_TRUE(bool(Serialized));
  ASSERT_TRUE(bool(ProtobufMarshaller.fromProtobuf(*Serialized)));

  // Relative path is absolute.
  *Serialized->mutable_canonical_declaration()->mutable_file_path() =
      convert_to_slash("/path/to/Declaration.h");
  Deserialized = ProtobufMarshaller.fromProtobuf(*Serialized);
  EXPECT_FALSE(bool(Deserialized));
  llvm::consumeError(Deserialized.takeError());

  // Fail with an invalid URI.
  Sym.Definition.FileURI = "Not A URI";
  Serialized = ProtobufMarshaller.toProtobuf(Sym);
  EXPECT_FALSE(bool(Serialized));
  llvm::consumeError(Serialized.takeError());

  // Schemes other than "file" can not be used.
  auto UnittestURI = URI::create(testPath("home/SomePath.h"), "unittest");
  ASSERT_TRUE(bool(UnittestURI));
  Sym.Definition.FileURI = Strings.save(UnittestURI->toString()).begin();
  Serialized = ProtobufMarshaller.toProtobuf(Sym);
  EXPECT_FALSE(bool(Serialized));
  llvm::consumeError(Serialized.takeError());

  // Passing root that is not prefix of the original file path.
  Sym.Definition.FileURI = testPathURI("home/File.h", Strings);
  // Check that the symbol is valid and passing the correct path works.
  Serialized = ProtobufMarshaller.toProtobuf(Sym);
  ASSERT_TRUE(bool(Serialized));
  Deserialized = ProtobufMarshaller.fromProtobuf(*Serialized);
  ASSERT_TRUE(bool(Deserialized));
  EXPECT_STREQ(Deserialized->Definition.FileURI,
               testPathURI("home/File.h", Strings));
  // Fail with a wrong root.
  Marshaller WrongMarshaller(testPath("nothome/"), testPath("home/"));
  Serialized = WrongMarshaller.toProtobuf(Sym);
  EXPECT_FALSE(Serialized);
  llvm::consumeError(Serialized.takeError());
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

  Marshaller ProtobufMarshaller(testPath("llvm-project/"),
                                testPath("llvm-project/"));

  auto Serialized = ProtobufMarshaller.toProtobuf(Ref);
  ASSERT_TRUE(bool(Serialized));
  auto Deserialized = ProtobufMarshaller.fromProtobuf(*Serialized);
  ASSERT_TRUE(bool(Deserialized));
  EXPECT_EQ(toYAML(Ref), toYAML(*Deserialized));
}

TEST(RemoteMarshallingTest, IncludeHeaderURIs) {
  llvm::BumpPtrAllocator Arena;
  llvm::UniqueStringSaver Strings(Arena);

  clangd::Symbol Sym = createSymbol("remote/", Strings);

  clangd::Symbol::IncludeHeaderWithReferences Header;
  // Add only valid headers.
  Header.IncludeHeader = Strings.save(
      URI::createFile("/usr/local/user/home/project/Header.h").toString());
  Header.References = 21;
  Sym.IncludeHeaders.push_back(Header);
  Header.IncludeHeader = Strings.save("<iostream>");
  Header.References = 100;
  Sym.IncludeHeaders.push_back(Header);
  Header.IncludeHeader = Strings.save("\"cstdio\"");
  Header.References = 200;
  Sym.IncludeHeaders.push_back(Header);

  Marshaller ProtobufMarshaller(convert_to_slash("/"), convert_to_slash("/"));

  auto Serialized = ProtobufMarshaller.toProtobuf(Sym);
  ASSERT_TRUE(bool(Serialized));
  EXPECT_EQ(static_cast<size_t>(Serialized->headers_size()),
            Sym.IncludeHeaders.size());
  auto Deserialized = ProtobufMarshaller.fromProtobuf(*Serialized);
  ASSERT_TRUE(bool(Deserialized));
  EXPECT_EQ(toYAML(Sym), toYAML(*Deserialized));

  // This is an absolute path to a header: can not be transmitted over the wire.
  Header.IncludeHeader = Strings.save(testPath("project/include/Common.h"));
  Header.References = 42;
  Sym.IncludeHeaders.push_back(Header);
  Serialized = ProtobufMarshaller.toProtobuf(Sym);
  EXPECT_FALSE(bool(Serialized));
  llvm::consumeError(Serialized.takeError());

  // Remove last invalid header.
  Sym.IncludeHeaders.pop_back();
  // This is not a valid header: can not be transmitted over the wire;
  Header.IncludeHeader = Strings.save("NotAHeader");
  Header.References = 5;
  Sym.IncludeHeaders.push_back(Header);
  Serialized = ProtobufMarshaller.toProtobuf(Sym);
  EXPECT_FALSE(bool(Serialized));
  llvm::consumeError(Serialized.takeError());

  // Try putting an invalid header into already serialized symbol.
  Sym.IncludeHeaders.pop_back();
  Serialized = ProtobufMarshaller.toProtobuf(Sym);
  ASSERT_TRUE(bool(Serialized));
  HeaderWithReferences InvalidHeader;
  InvalidHeader.set_header(convert_to_slash("/absolute/path/Header.h"));
  InvalidHeader.set_references(9000);
  *Serialized->add_headers() = InvalidHeader;
  Deserialized = ProtobufMarshaller.fromProtobuf(*Serialized);
  EXPECT_FALSE(bool(Deserialized));
  llvm::consumeError(Deserialized.takeError());
}

TEST(RemoteMarshallingTest, LookupRequestSerialization) {
  clangd::LookupRequest Request;
  Request.IDs.insert(llvm::cantFail(SymbolID::fromStr("0000000000000001")));
  Request.IDs.insert(llvm::cantFail(SymbolID::fromStr("0000000000000002")));

  Marshaller ProtobufMarshaller(testPath("remote/"), testPath("local/"));

  auto Serialized = ProtobufMarshaller.toProtobuf(Request);
  EXPECT_EQ(static_cast<unsigned>(Serialized.ids_size()), Request.IDs.size());
  auto Deserialized = ProtobufMarshaller.fromProtobuf(&Serialized);
  ASSERT_TRUE(bool(Deserialized));
  EXPECT_EQ(Deserialized->IDs, Request.IDs);
}

TEST(RemoteMarshallingTest, LookupRequestFailingSerialization) {
  clangd::LookupRequest Request;
  Marshaller ProtobufMarshaller(testPath("remote/"), testPath("local/"));
  auto Serialized = ProtobufMarshaller.toProtobuf(Request);
  Serialized.add_ids("Invalid Symbol ID");
  auto Deserialized = ProtobufMarshaller.fromProtobuf(&Serialized);
  EXPECT_FALSE(bool(Deserialized));
  llvm::consumeError(Deserialized.takeError());
}

TEST(RemoteMarshallingTest, FuzzyFindRequestSerialization) {
  clangd::FuzzyFindRequest Request;
  Request.ProximityPaths = {testPath("local/Header.h"),
                            testPath("local/subdir/OtherHeader.h"),
                            testPath("remote/File.h"), "Not a Path."};
  Marshaller ProtobufMarshaller(testPath("remote/"), testPath("local/"));
  auto Serialized = ProtobufMarshaller.toProtobuf(Request);
  EXPECT_EQ(Serialized.proximity_paths_size(), 2);
  auto Deserialized = ProtobufMarshaller.fromProtobuf(&Serialized);
  ASSERT_TRUE(bool(Deserialized));
  EXPECT_THAT(Deserialized->ProximityPaths,
              testing::ElementsAre(testPath("remote/Header.h"),
                                   testPath("remote/subdir/OtherHeader.h")));
}

TEST(RemoteMarshallingTest, RefsRequestSerialization) {
  clangd::RefsRequest Request;
  Request.IDs.insert(llvm::cantFail(SymbolID::fromStr("0000000000000001")));
  Request.IDs.insert(llvm::cantFail(SymbolID::fromStr("0000000000000002")));

  Request.Limit = 9000;
  Request.Filter = RefKind::Spelled | RefKind::Declaration;

  Marshaller ProtobufMarshaller(testPath("remote/"), testPath("local/"));

  auto Serialized = ProtobufMarshaller.toProtobuf(Request);
  EXPECT_EQ(static_cast<unsigned>(Serialized.ids_size()), Request.IDs.size());
  EXPECT_EQ(Serialized.limit(), Request.Limit);
  auto Deserialized = ProtobufMarshaller.fromProtobuf(&Serialized);
  ASSERT_TRUE(bool(Deserialized));
  EXPECT_EQ(Deserialized->IDs, Request.IDs);
  ASSERT_TRUE(Deserialized->Limit);
  EXPECT_EQ(*Deserialized->Limit, Request.Limit);
  EXPECT_EQ(Deserialized->Filter, Request.Filter);
}

TEST(RemoteMarshallingTest, RefsRequestFailingSerialization) {
  clangd::RefsRequest Request;
  Marshaller ProtobufMarshaller(testPath("remote/"), testPath("local/"));
  auto Serialized = ProtobufMarshaller.toProtobuf(Request);
  Serialized.add_ids("Invalid Symbol ID");
  auto Deserialized = ProtobufMarshaller.fromProtobuf(&Serialized);
  EXPECT_FALSE(bool(Deserialized));
  llvm::consumeError(Deserialized.takeError());
}

TEST(RemoteMarshallingTest, RelationsRequestSerialization) {
  clangd::RelationsRequest Request;
  Request.Subjects.insert(
      llvm::cantFail(SymbolID::fromStr("0000000000000001")));
  Request.Subjects.insert(
      llvm::cantFail(SymbolID::fromStr("0000000000000002")));

  Request.Limit = 9000;
  Request.Predicate = RelationKind::BaseOf;

  Marshaller ProtobufMarshaller(testPath("remote/"), testPath("local/"));

  auto Serialized = ProtobufMarshaller.toProtobuf(Request);
  EXPECT_EQ(static_cast<unsigned>(Serialized.subjects_size()),
            Request.Subjects.size());
  EXPECT_EQ(Serialized.limit(), Request.Limit);
  EXPECT_EQ(static_cast<RelationKind>(Serialized.predicate()),
            Request.Predicate);
  auto Deserialized = ProtobufMarshaller.fromProtobuf(&Serialized);
  ASSERT_TRUE(bool(Deserialized));
  EXPECT_EQ(Deserialized->Subjects, Request.Subjects);
  ASSERT_TRUE(Deserialized->Limit);
  EXPECT_EQ(*Deserialized->Limit, Request.Limit);
  EXPECT_EQ(Deserialized->Predicate, Request.Predicate);
}

TEST(RemoteMarshallingTest, RelationsRequestFailingSerialization) {
  RelationsRequest Serialized;
  Serialized.add_subjects("ZZZZZZZZZZZZZZZZ");
  Marshaller ProtobufMarshaller(testPath("remote/"), testPath("local/"));
  auto Deserialized = ProtobufMarshaller.fromProtobuf(&Serialized);
  EXPECT_FALSE(bool(Deserialized));
  llvm::consumeError(Deserialized.takeError());
}

TEST(RemoteMarshallingTest, RelationsSerializion) {
  llvm::BumpPtrAllocator Arena;
  llvm::UniqueStringSaver Strings(Arena);

  clangd::Symbol Sym = createSymbol("remote/", Strings);
  SymbolID ID = llvm::cantFail(SymbolID::fromStr("0000000000000002"));
  Marshaller ProtobufMarshaller(testPath("remote/"), testPath("local/"));
  auto Serialized = ProtobufMarshaller.toProtobuf(ID, Sym);
  ASSERT_TRUE(bool(Serialized));
  auto Deserialized = ProtobufMarshaller.fromProtobuf(*Serialized);
  ASSERT_TRUE(bool(Deserialized));
  EXPECT_THAT(Deserialized->first, ID);
  EXPECT_THAT(Deserialized->second.ID, Sym.ID);
}

TEST(RemoteMarshallingTest, RelativePathToURITranslation) {
  Marshaller ProtobufMarshaller(/*RemoteIndexRoot=*/"",
                                /*LocalIndexRoot=*/testPath("home/project/"));
  auto URIString = ProtobufMarshaller.relativePathToURI("lib/File.cpp");
  ASSERT_TRUE(bool(URIString));
  // RelativePath can not be absolute.
  URIString = ProtobufMarshaller.relativePathToURI("/lib/File.cpp");
  EXPECT_FALSE(bool(URIString));
  llvm::consumeError(URIString.takeError());
  // RelativePath can not be empty.
  URIString = ProtobufMarshaller.relativePathToURI(std::string());
  EXPECT_FALSE(bool(URIString));
  llvm::consumeError(URIString.takeError());
}

TEST(RemoteMarshallingTest, URIToRelativePathTranslation) {
  llvm::BumpPtrAllocator Arena;
  llvm::UniqueStringSaver Strings(Arena);
  Marshaller ProtobufMarshaller(/*RemoteIndexRoot=*/testPath("remote/project/"),
                                /*LocalIndexRoot=*/"");
  auto RelativePath = ProtobufMarshaller.uriToRelativePath(
      testPathURI("remote/project/lib/File.cpp", Strings));
  ASSERT_TRUE(bool(RelativePath));
  // RemoteIndexRoot has to be be a prefix of the file path.
  Marshaller WrongMarshaller(
      /*RemoteIndexRoot=*/testPath("remote/other/project/"),
      /*LocalIndexRoot=*/"");
  RelativePath = WrongMarshaller.uriToRelativePath(
      testPathURI("remote/project/lib/File.cpp", Strings));
  EXPECT_FALSE(bool(RelativePath));
  llvm::consumeError(RelativePath.takeError());
}

} // namespace
} // namespace remote
} // namespace clangd
} // namespace clang
