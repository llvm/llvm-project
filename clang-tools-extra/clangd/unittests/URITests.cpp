//===-- URITests.cpp  ---------------------------------*- C++ -*-----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Matchers.h"
#include "Protocol.h"
#include "TestFS.h"
#include "URI.h"
#include "index/PathIdentity.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {

// Force the unittest URI scheme to be linked,
[[maybe_unused]] static int UnittestSchemeAnchorDest =
    UnittestSchemeAnchorSource;

namespace {

using ::testing::AllOf;

MATCHER_P(scheme, S, "") { return arg.scheme() == S; }
MATCHER_P(authority, A, "") { return arg.authority() == A; }
MATCHER_P(body, B, "") { return arg.body() == B; }

std::string createOrDie(llvm::StringRef AbsolutePath,
                        llvm::StringRef Scheme = "file") {
  auto Uri = URI::create(AbsolutePath, Scheme);
  if (!Uri)
    llvm_unreachable(toString(Uri.takeError()).c_str());
  return Uri->toString();
}

URI parseOrDie(llvm::StringRef Uri) {
  auto U = URI::parse(Uri);
  if (!U)
    llvm_unreachable(toString(U.takeError()).c_str());
  return *U;
}

TEST(PercentEncodingTest, Encode) {
  EXPECT_EQ(URI("x", /*authority=*/"", "a/b/c").toString(), "x:a/b/c");
  EXPECT_EQ(URI("x", /*authority=*/"", "a!b;c~").toString(), "x:a%21b%3Bc~");
  EXPECT_EQ(URI("x", /*authority=*/"", "a123b").toString(), "x:a123b");
  EXPECT_EQ(URI("x", /*authority=*/"", "a:b;c").toString(), "x:a:b%3Bc");
}

TEST(PercentEncodingTest, Decode) {
  EXPECT_EQ(parseOrDie("x:a/b/c").body(), "a/b/c");

  EXPECT_EQ(parseOrDie("s%2b://%3a/%3").scheme(), "s+");
  EXPECT_EQ(parseOrDie("s%2b://%3a/%3").authority(), ":");
  EXPECT_EQ(parseOrDie("s%2b://%3a/%3").body(), "/%3");

  EXPECT_EQ(parseOrDie("x:a%21b%3ac~").body(), "a!b:c~");
  EXPECT_EQ(parseOrDie("x:a:b%3bc").body(), "a:b;c");
}

std::string resolveOrDie(const URI &U, llvm::StringRef HintPath = "") {
  auto Path = URI::resolve(U, HintPath);
  if (!Path)
    llvm_unreachable(toString(Path.takeError()).c_str());
  return *Path;
}

TEST(URITest, Create) {
#ifdef _WIN32
  EXPECT_THAT(createOrDie("c:\\x\\y\\z"), "file:///c:/x/y/z");
#else
  EXPECT_THAT(createOrDie("/x/y/z"), "file:///x/y/z");
  EXPECT_THAT(createOrDie("/(x)/y/\\ z"), "file:///%28x%29/y/%5C%20z");
#endif
}

TEST(URITest, CreateUNC) {
#ifdef _WIN32
  EXPECT_THAT(createOrDie("\\\\test.org\\x\\y\\z"), "file://test.org/x/y/z");
  EXPECT_THAT(createOrDie("\\\\10.0.0.1\\x\\y\\z"), "file://10.0.0.1/x/y/z");
#else
  EXPECT_THAT(createOrDie("//test.org/x/y/z"), "file://test.org/x/y/z");
  EXPECT_THAT(createOrDie("//10.0.0.1/x/y/z"), "file://10.0.0.1/x/y/z");
#endif
}

TEST(URITest, FailedCreate) {
  EXPECT_ERROR(URI::create("/x/y/z", "no"));
  // Path has to be absolute.
  EXPECT_ERROR(URI::create("x/y/z", "file"));
}

TEST(URITest, Parse) {
  EXPECT_THAT(parseOrDie("file://auth/x/y/z"),
              AllOf(scheme("file"), authority("auth"), body("/x/y/z")));

  EXPECT_THAT(parseOrDie("file://au%3dth/%28x%29/y/%5c%20z"),
              AllOf(scheme("file"), authority("au=th"), body("/(x)/y/\\ z")));

  EXPECT_THAT(parseOrDie("file:///%28x%29/y/%5c%20z"),
              AllOf(scheme("file"), authority(""), body("/(x)/y/\\ z")));
  EXPECT_THAT(parseOrDie("file:///x/y/z"),
              AllOf(scheme("file"), authority(""), body("/x/y/z")));
  EXPECT_THAT(parseOrDie("file:"),
              AllOf(scheme("file"), authority(""), body("")));
  EXPECT_THAT(parseOrDie("file:///x/y/z%2"),
              AllOf(scheme("file"), authority(""), body("/x/y/z%2")));
  EXPECT_THAT(parseOrDie("http://llvm.org"),
              AllOf(scheme("http"), authority("llvm.org"), body("")));
  EXPECT_THAT(parseOrDie("http://llvm.org/"),
              AllOf(scheme("http"), authority("llvm.org"), body("/")));
  EXPECT_THAT(parseOrDie("http://llvm.org/D"),
              AllOf(scheme("http"), authority("llvm.org"), body("/D")));
  EXPECT_THAT(parseOrDie("http:/"),
              AllOf(scheme("http"), authority(""), body("/")));
  EXPECT_THAT(parseOrDie("urn:isbn:0451450523"),
              AllOf(scheme("urn"), authority(""), body("isbn:0451450523")));
  EXPECT_THAT(
      parseOrDie("file:///c:/windows/system32/"),
      AllOf(scheme("file"), authority(""), body("/c:/windows/system32/")));
}

TEST(URITest, ParseFailed) {
  // Expect ':' in URI.
  EXPECT_ERROR(URI::parse("file//x/y/z"));
  // Empty.
  EXPECT_ERROR(URI::parse(""));
  EXPECT_ERROR(URI::parse(":/a/b/c"));
  EXPECT_ERROR(URI::parse("\"/a/b/c\" IWYU pragma: abc"));
}

TEST(URITest, Resolve) {
#ifdef _WIN32
  // Expected path style depends on LLVM_WINDOWS_PREFER_FORWARD_SLASH.
  EXPECT_THAT(resolveOrDie(parseOrDie("file:///c%3a/x/y/z")),
              llvm::sys::path::native("c:/x/y/z"));
  EXPECT_THAT(resolveOrDie(parseOrDie("file:///c:/x/y/z")),
              llvm::sys::path::native("c:/x/y/z"));
#else
  EXPECT_EQ(resolveOrDie(parseOrDie("file:/a/b/c")), "/a/b/c");
  EXPECT_EQ(resolveOrDie(parseOrDie("file://auth/a/b/c")), "//auth/a/b/c");
  EXPECT_THAT(resolveOrDie(parseOrDie("file://au%3dth/%28x%29/y/%20z")),
              "//au=th/(x)/y/ z");
  EXPECT_THAT(resolveOrDie(parseOrDie("file:///c:/x/y/z")), "c:/x/y/z");
#endif
  EXPECT_EQ(resolveOrDie(parseOrDie("unittest:///a"), testPath("x")),
            testPath("a"));
}

TEST(URITest, IndexFileIdentityDriveLetter) {
  EXPECT_EQ(indexFileIdentity("file:///C:/proj/a.cpp"),
            indexFileIdentity("file:///c:/proj/a.cpp"));
  EXPECT_EQ(indexFileIdentity("file:///C:/proj/a.cpp"),
            indexFileIdentity("C:/proj/a.cpp"));
  EXPECT_EQ(indexFileIdentity("file:///C:/proj/a.cpp"),
            indexFileIdentity("c:\\proj\\a.cpp"));
  EXPECT_NE(indexFileIdentity("file:///C:/proj/a.cpp"),
            indexFileIdentity("file:///D:/proj/a.cpp"));
}

TEST(URITest, IndexFileIdentityPreservesCase) {
  EXPECT_NE(indexFileIdentity("file:///C:/proj/Foo.h"),
            indexFileIdentity("file:///C:/proj/foo.h"));
  EXPECT_NE(indexFileIdentity("file:///proj/Foo.h"),
            indexFileIdentity("file:///proj/foo.h"));
  EXPECT_EQ(indexFileIdentity("file:///C:/proj/%46oo.h"),
            indexFileIdentity("c:\\proj\\Foo.h"));
}

TEST(URITest, IndexFileIdentityRejectsInvalidURI) {
  EXPECT_FALSE(indexFileIdentity("file:relative/a.cpp"));
  EXPECT_FALSE(indexFileIdentity("file://server"));

  // Strings without URI syntax remain valid path keys.
  EXPECT_EQ(indexFileIdentity("relative/a.cpp"),
            indexFileIdentityFrom(Path("relative/a.cpp")));
#ifndef CLANGD_PATH_CASE_INSENSITIVE
  EXPECT_NE(indexFileIdentity("C:notes"), indexFileIdentity("c:notes"));
#endif
}

TEST(URITest, IndexFileIdentityOpaqueURI) {
  auto Key = indexFileIdentity("unknown-scheme:///proj/a.cpp");
  ASSERT_TRUE(Key);
  EXPECT_EQ(Key->raw(), "unknown-scheme:///proj/a.cpp");
  EXPECT_NE(Key, indexFileIdentity("unknown-scheme:///proj/A.cpp"));
  EXPECT_NE(Key, indexFileIdentityFrom(Path(Key->raw().str())));
}

TEST(URITest, IndexFileIdentityBorrowedLookup) {
  for (const char *U : {"file:/a/b", "file:///a/b", "file://server/share/a",
                        "file:////server/share/a", "file:///C:/proj/a.cpp",
                        "file:///c%3A/proj/a%20b.cpp", "file:///",
                        "file:/C:", "file:///tmp/a\\b", "file:///tmp/%61\\b"}) {
    SCOPED_TRACE(U);
    auto Resolved = resolveOrDie(parseOrDie(U));
    auto Owned = indexFileIdentityFrom(Path(Resolved));
    llvm::SmallString<256> Storage;
    auto Borrowed = indexFileIdentity(U, Storage);
    ASSERT_TRUE(Borrowed);
    ASSERT_TRUE(Owned);
    EXPECT_TRUE(IndexFileKeyInfo::isEqual(*Borrowed, *Owned));
    EXPECT_EQ(IndexFileKeyInfo::getHashValue(*Borrowed),
              IndexFileKeyInfo::getHashValue(*Owned));
    IndexFileSet Files;
    Files.insert(*Owned);
    EXPECT_NE(Files.find_as(*Borrowed), Files.end());
    if (!llvm::StringRef(U).contains('%') &&
        !llvm::StringRef(U).contains('\\')) {
      EXPECT_TRUE(Storage.empty());
      EXPECT_GE(Borrowed->Value.data(), U);
      EXPECT_LE(Borrowed->Value.end(), U + strlen(U));
    }
  }
}

TEST(URITest, URIForFileDriveLetter) {
  auto Upper = URIForFile::fromURI(parseOrDie("file:///C:/proj/a.cpp"), "");
  auto Lower = URIForFile::fromURI(parseOrDie("file:///c:/proj/a.cpp"), "");
  ASSERT_TRUE(bool(Upper)) << Upper.takeError();
  ASSERT_TRUE(bool(Lower)) << Lower.takeError();
  EXPECT_EQ(*Upper, *Lower);
  EXPECT_FALSE(*Lower < *Upper);
  EXPECT_FALSE(*Upper < *Lower);
}

TEST(URITest, URIForFilePreservesCase) {
  auto Upper = URIForFile::fromURI(parseOrDie("file:///C:/proj/Foo.h"), "");
  auto Lower = URIForFile::fromURI(parseOrDie("file:///C:/proj/foo.h"), "");
  auto Alias = URIForFile::fromURI(parseOrDie("file:///c:/proj/Foo.h"), "");
  ASSERT_TRUE(bool(Upper)) << Upper.takeError();
  ASSERT_TRUE(bool(Lower)) << Lower.takeError();
  ASSERT_TRUE(bool(Alias)) << Alias.takeError();
  EXPECT_NE(*Upper, *Lower);
  EXPECT_EQ(*Upper, *Alias);
  EXPECT_TRUE(*Upper < *Lower);
  EXPECT_TRUE(*Alias < *Lower);
  EXPECT_FALSE(*Lower < *Upper);
  EXPECT_FALSE(*Upper < *Alias);
  EXPECT_FALSE(*Alias < *Upper);
}

TEST(URITest, ResolveUNC) {
#ifdef _WIN32
  // Expected path style depends on LLVM_WINDOWS_PREFER_FORWARD_SLASH.
  EXPECT_THAT(resolveOrDie(parseOrDie("file://example.com/x/y/z")),
              llvm::sys::path::native("//example.com/x/y/z"));
  EXPECT_THAT(resolveOrDie(parseOrDie("file://127.0.0.1/x/y/z")),
              llvm::sys::path::native("//127.0.0.1/x/y/z"));
  // Ensure non-traditional file URI still resolves to correct UNC path.
  EXPECT_THAT(resolveOrDie(parseOrDie("file:////127.0.0.1/x/y/z")),
              llvm::sys::path::native("//127.0.0.1/x/y/z"));
#else
  EXPECT_THAT(resolveOrDie(parseOrDie("file://example.com/x/y/z")),
              "//example.com/x/y/z");
  EXPECT_THAT(resolveOrDie(parseOrDie("file://127.0.0.1/x/y/z")),
              "//127.0.0.1/x/y/z");
#endif
}

std::string resolvePathOrDie(llvm::StringRef AbsPath,
                             llvm::StringRef HintPath = "") {
  auto Path = URI::resolvePath(AbsPath, HintPath);
  if (!Path)
    llvm_unreachable(toString(Path.takeError()).c_str());
  return *Path;
}

TEST(URITest, ResolvePath) {
  StringRef FilePath =
#ifdef _WIN32
      "c:\\x\\y\\z";
#else
      "/a/b/c";
#endif
  EXPECT_EQ(resolvePathOrDie(FilePath), FilePath);
  EXPECT_EQ(resolvePathOrDie(testPath("x"), testPath("hint")), testPath("x"));
  // HintPath is not in testRoot(); resolution fails.
  auto Resolve = URI::resolvePath(testPath("x"), FilePath);
  EXPECT_FALSE(Resolve);
  llvm::consumeError(Resolve.takeError());
}

TEST(URITest, Platform) {
  auto Path = testPath("x");
  auto U = URI::create(Path, "file");
  EXPECT_TRUE(static_cast<bool>(U));
  EXPECT_THAT(resolveOrDie(*U), Path);
}

TEST(URITest, ResolveFailed) {
  auto FailedResolve = [](StringRef Uri) {
    auto Path = URI::resolve(parseOrDie(Uri));
    if (!Path) {
      consumeError(Path.takeError());
      return true;
    }
    return false;
  };

  // Invalid scheme.
  EXPECT_TRUE(FailedResolve("no:/a/b/c"));
  // File path needs to be absolute.
  EXPECT_TRUE(FailedResolve("file:a/b/c"));
}

} // namespace
} // namespace clangd
} // namespace clang
