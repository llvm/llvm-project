//===-- PathMappingTests.cpp  ------------------------*- C++ -*-----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PathMapping.h"
#include "llvm/Support/JSON.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <string>
namespace clang {
namespace clangd {
namespace {
using ::testing::ElementsAre;
MATCHER_P2(Mapping, ClientPath, ServerPath, "") {
  return arg.ClientPath == ClientPath && arg.ServerPath == ServerPath;
}

bool failedParse(llvm::StringRef RawMappings) {
  llvm::Expected<PathMappings> Mappings = parsePathMappings(RawMappings);
  if (!Mappings) {
    consumeError(Mappings.takeError());
    return true;
  }
  return false;
}

TEST(ParsePathMappingTests, WindowsPath) {
  // Relative path to C drive
  EXPECT_TRUE(failedParse(R"(C:a=/root)"));
  EXPECT_TRUE(failedParse(R"(\C:a=/root)"));
  // Relative path to current drive.
  EXPECT_TRUE(failedParse(R"(\a=/root)"));
  // Absolute paths
  llvm::Expected<PathMappings> ParsedMappings =
      parsePathMappings(R"(C:\a=/root)");
  ASSERT_TRUE(bool(ParsedMappings));
  EXPECT_THAT(*ParsedMappings, ElementsAre(Mapping("/C:/a", "/root")));
  // Absolute UNC path
  ParsedMappings = parsePathMappings(R"(\\Server\C$=/root)");
  ASSERT_TRUE(bool(ParsedMappings));
  EXPECT_THAT(*ParsedMappings, ElementsAre(Mapping("//Server/C$", "/root")));
}

TEST(ParsePathMappingTests, UnixPath) {
  // Relative unix path
  EXPECT_TRUE(failedParse("a/b=/root"));
  // Absolute unix path
  llvm::Expected<PathMappings> ParsedMappings = parsePathMappings("/A/b=/root");
  ASSERT_TRUE(bool(ParsedMappings));
  EXPECT_THAT(*ParsedMappings, ElementsAre(Mapping("/A/b", "/root")));
  // Absolute unix path w/ backslash
  ParsedMappings = parsePathMappings(R"(/a/b\\ar=/root)");
  ASSERT_TRUE(bool(ParsedMappings));
  EXPECT_THAT(*ParsedMappings, ElementsAre(Mapping(R"(/a/b\\ar)", "/root")));
}

TEST(ParsePathMappingTests, ImproperFormat) {
  // uneven mappings
  EXPECT_TRUE(failedParse("/home/myuser1="));
  // mappings need to be absolute
  EXPECT_TRUE(failedParse("home/project=/workarea/project"));
  // duplicate delimiter
  EXPECT_TRUE(failedParse("/home==/workarea"));
  // no delimiter
  EXPECT_TRUE(failedParse("/home"));
  // improper delimiter
  EXPECT_TRUE(failedParse("/home,/workarea"));
}

TEST(ParsePathMappingTests, ParsesMultiple) {
  std::string RawPathMappings =
      "/home/project=/workarea/project,/home/project/.includes=/opt/include";
  auto Parsed = parsePathMappings(RawPathMappings);
  ASSERT_TRUE(bool(Parsed));
  EXPECT_THAT(*Parsed,
              ElementsAre(Mapping("/home/project", "/workarea/project"),
                          Mapping("/home/project/.includes", "/opt/include")));
}

bool mapsProperly(llvm::StringRef Orig, llvm::StringRef Expected,
                  llvm::StringRef RawMappings, PathMapping::Direction Dir) {
  llvm::Expected<PathMappings> Mappings = parsePathMappings(RawMappings);
  if (!Mappings)
    return false;
  llvm::Optional<std::string> MappedPath = doPathMapping(Orig, Dir, *Mappings);
  std::string Actual = MappedPath ? *MappedPath : Orig.str();
  EXPECT_STREQ(Expected.str().c_str(), Actual.c_str());
  return Expected == Actual;
}

TEST(DoPathMappingTests, PreservesOriginal) {
  // Preserves original path when no mapping
  EXPECT_TRUE(mapsProperly("file:///home", "file:///home", "",
                           PathMapping::Direction::ClientToServer));
}

TEST(DoPathMappingTests, UsesFirstMatch) {
  EXPECT_TRUE(mapsProperly("file:///home/foo.cpp", "file:///workarea1/foo.cpp",
                           "/home=/workarea1,/home=/workarea2",
                           PathMapping::Direction::ClientToServer));
}

TEST(DoPathMappingTests, IgnoresSubstrings) {
  // Doesn't map substrings that aren't a proper path prefix
  EXPECT_TRUE(mapsProperly("file://home/foo-bar.cpp", "file://home/foo-bar.cpp",
                           "/home/foo=/home/bar",
                           PathMapping::Direction::ClientToServer));
}

TEST(DoPathMappingTests, MapsOutgoingPaths) {
  // When IsIncoming is false (i.e.a  response), map the other way
  EXPECT_TRUE(mapsProperly("file:///workarea/foo.cpp", "file:///home/foo.cpp",
                           "/home=/workarea",
                           PathMapping::Direction::ServerToClient));
}

TEST(DoPathMappingTests, OnlyMapFileUris) {
  EXPECT_TRUE(mapsProperly("test:///home/foo.cpp", "test:///home/foo.cpp",
                           "/home=/workarea",
                           PathMapping::Direction::ClientToServer));
}

TEST(DoPathMappingTests, RespectsCaseSensitivity) {
  EXPECT_TRUE(mapsProperly("file:///HOME/foo.cpp", "file:///HOME/foo.cpp",
                           "/home=/workarea",
                           PathMapping::Direction::ClientToServer));
}

TEST(DoPathMappingTests, MapsWindowsPaths) {
  // Maps windows properly
  EXPECT_TRUE(mapsProperly("file:///C:/home/foo.cpp",
                           "file:///C:/workarea/foo.cpp", R"(C:\home=C:\workarea)",
                           PathMapping::Direction::ClientToServer));
}

TEST(DoPathMappingTests, MapsWindowsUnixInterop) {
  // Path mappings with a windows-style client path and unix-style server path
  EXPECT_TRUE(mapsProperly(
      "file:///C:/home/foo.cpp", "file:///workarea/foo.cpp",
      R"(C:\home=/workarea)", PathMapping::Direction::ClientToServer));
}

TEST(ApplyPathMappingTests, PreservesOriginalParams) {
  auto Params = llvm::json::parse(R"({
    "textDocument": {"uri": "file:///home/foo.cpp"},
    "position": {"line": 0, "character": 0}
  })");
  ASSERT_TRUE(bool(Params));
  llvm::json::Value ExpectedParams = *Params;
  PathMappings Mappings;
  applyPathMappings(*Params, PathMapping::Direction::ClientToServer, Mappings);
  EXPECT_EQ(*Params, ExpectedParams);
}

TEST(ApplyPathMappingTests, MapsAllMatchingPaths) {
  // Handles nested objects and array values
  auto Params = llvm::json::parse(R"({
    "rootUri": {"uri": "file:///home/foo.cpp"},
    "workspaceFolders": ["file:///home/src", "file:///tmp"]
  })");
  auto ExpectedParams = llvm::json::parse(R"({
    "rootUri": {"uri": "file:///workarea/foo.cpp"},
    "workspaceFolders": ["file:///workarea/src", "file:///tmp"]
  })");
  auto Mappings = parsePathMappings("/home=/workarea");
  ASSERT_TRUE(bool(Params) && bool(ExpectedParams) && bool(Mappings));
  applyPathMappings(*Params, PathMapping::Direction::ClientToServer, *Mappings);
  EXPECT_EQ(*Params, *ExpectedParams);
}

TEST(ApplyPathMappingTests, MapsOutbound) {
  auto Params = llvm::json::parse(R"({
    "id": 1,
    "result": [
      {"uri": "file:///opt/include/foo.h"},
      {"uri": "file:///workarea/src/foo.cpp"}]
  })");
  auto ExpectedParams = llvm::json::parse(R"({
    "id": 1,
    "result": [
      {"uri": "file:///home/.includes/foo.h"},
      {"uri": "file:///home/src/foo.cpp"}]
  })");
  auto Mappings =
      parsePathMappings("/home=/workarea,/home/.includes=/opt/include");
  ASSERT_TRUE(bool(Params) && bool(ExpectedParams) && bool(Mappings));
  applyPathMappings(*Params, PathMapping::Direction::ServerToClient, *Mappings);
  EXPECT_EQ(*Params, *ExpectedParams);
}

TEST(ApplyPathMappingTests, MapsKeys) {
  auto Params = llvm::json::parse(R"({
    "changes": {
      "file:///home/foo.cpp": {"newText": "..."},
      "file:///home/src/bar.cpp": {"newText": "..."}
    }
  })");
  auto ExpectedParams = llvm::json::parse(R"({
    "changes": {
      "file:///workarea/foo.cpp": {"newText": "..."},
      "file:///workarea/src/bar.cpp": {"newText": "..."}
    }
  })");
  auto Mappings = parsePathMappings("/home=/workarea");
  ASSERT_TRUE(bool(Params) && bool(ExpectedParams) && bool(Mappings));
  applyPathMappings(*Params, PathMapping::Direction::ClientToServer, *Mappings);
  EXPECT_EQ(*Params, *ExpectedParams);
}

} // namespace
} // namespace clangd
} // namespace clang
