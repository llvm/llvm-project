//===-- URITests.cpp  ---------------------------------*- C++ -*-----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Matchers.h"
#include "TestFS.h"
#include "URI.h"
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
  EXPECT_THAT(resolveOrDie(parseOrDie("file:///c%3a/x/y/z")), "c:\\x\\y\\z");
  EXPECT_THAT(resolveOrDie(parseOrDie("file:///c:/x/y/z")), "c:\\x\\y\\z");
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

TEST(URITest, ResolveUNC) {
#ifdef _WIN32
  EXPECT_THAT(resolveOrDie(parseOrDie("file://example.com/x/y/z")),
              "\\\\example.com\\x\\y\\z");
  EXPECT_THAT(resolveOrDie(parseOrDie("file://127.0.0.1/x/y/z")),
              "\\\\127.0.0.1\\x\\y\\z");
  // Ensure non-traditional file URI still resolves to correct UNC path.
  EXPECT_THAT(resolveOrDie(parseOrDie("file:////127.0.0.1/x/y/z")),
              "\\\\127.0.0.1\\x\\y\\z");
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
