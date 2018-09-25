//===-- URITests.cpp  ---------------------------------*- C++ -*-----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "TestFS.h"
#include "URI.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {

// Force the unittest URI scheme to be linked,
static int LLVM_ATTRIBUTE_UNUSED UnittestSchemeAnchorDest =
    UnittestSchemeAnchorSource;

namespace {

using ::testing::AllOf;

MATCHER_P(Scheme, S, "") { return arg.scheme() == S; }
MATCHER_P(Authority, A, "") { return arg.authority() == A; }
MATCHER_P(Body, B, "") { return arg.body() == B; }

std::string createOrDie(llvm::StringRef AbsolutePath,
                        llvm::StringRef Scheme = "file") {
  auto Uri = URI::create(AbsolutePath, Scheme);
  if (!Uri)
    llvm_unreachable(llvm::toString(Uri.takeError()).c_str());
  return Uri->toString();
}

URI parseOrDie(llvm::StringRef Uri) {
  auto U = URI::parse(Uri);
  if (!U)
    llvm_unreachable(llvm::toString(U.takeError()).c_str());
  return *U;
}

TEST(PercentEncodingTest, Encode) {
  EXPECT_EQ(URI("x", /*Authority=*/"", "a/b/c").toString(), "x:a/b/c");
  EXPECT_EQ(URI("x", /*Authority=*/"", "a!b;c~").toString(), "x:a%21b%3bc~");
  EXPECT_EQ(URI("x", /*Authority=*/"", "a123b").toString(), "x:a123b");
}

TEST(PercentEncodingTest, Decode) {
  EXPECT_EQ(parseOrDie("x:a/b/c").body(), "a/b/c");

  EXPECT_EQ(parseOrDie("s%2b://%3a/%3").scheme(), "s+");
  EXPECT_EQ(parseOrDie("s%2b://%3a/%3").authority(), ":");
  EXPECT_EQ(parseOrDie("s%2b://%3a/%3").body(), "/%3");

  EXPECT_EQ(parseOrDie("x:a%21b%3ac~").body(), "a!b:c~");
}

std::string resolveOrDie(const URI &U, llvm::StringRef HintPath = "") {
  auto Path = URI::resolve(U, HintPath);
  if (!Path)
    llvm_unreachable(llvm::toString(Path.takeError()).c_str());
  return *Path;
}

TEST(URITest, Create) {
#ifdef _WIN32
  EXPECT_THAT(createOrDie("c:\\x\\y\\z"), "file:///c%3a/x/y/z");
#else
  EXPECT_THAT(createOrDie("/x/y/z"), "file:///x/y/z");
  EXPECT_THAT(createOrDie("/(x)/y/\\ z"), "file:///%28x%29/y/%5c%20z");
#endif
}

TEST(URITest, FailedCreate) {
  auto Fail = [](llvm::Expected<URI> U) {
    if (!U) {
      llvm::consumeError(U.takeError());
      return true;
    }
    return false;
  };
  EXPECT_TRUE(Fail(URI::create("/x/y/z", "no")));
  // Path has to be absolute.
  EXPECT_TRUE(Fail(URI::create("x/y/z", "file")));
}

TEST(URITest, Parse) {
  EXPECT_THAT(parseOrDie("file://auth/x/y/z"),
              AllOf(Scheme("file"), Authority("auth"), Body("/x/y/z")));

  EXPECT_THAT(parseOrDie("file://au%3dth/%28x%29/y/%5c%20z"),
              AllOf(Scheme("file"), Authority("au=th"), Body("/(x)/y/\\ z")));

  EXPECT_THAT(parseOrDie("file:///%28x%29/y/%5c%20z"),
              AllOf(Scheme("file"), Authority(""), Body("/(x)/y/\\ z")));
  EXPECT_THAT(parseOrDie("file:///x/y/z"),
              AllOf(Scheme("file"), Authority(""), Body("/x/y/z")));
  EXPECT_THAT(parseOrDie("file:"),
              AllOf(Scheme("file"), Authority(""), Body("")));
  EXPECT_THAT(parseOrDie("file:///x/y/z%2"),
              AllOf(Scheme("file"), Authority(""), Body("/x/y/z%2")));
  EXPECT_THAT(parseOrDie("http://llvm.org"),
              AllOf(Scheme("http"), Authority("llvm.org"), Body("")));
  EXPECT_THAT(parseOrDie("http://llvm.org/"),
              AllOf(Scheme("http"), Authority("llvm.org"), Body("/")));
  EXPECT_THAT(parseOrDie("http://llvm.org/D"),
              AllOf(Scheme("http"), Authority("llvm.org"), Body("/D")));
  EXPECT_THAT(parseOrDie("http:/"),
              AllOf(Scheme("http"), Authority(""), Body("/")));
  EXPECT_THAT(parseOrDie("urn:isbn:0451450523"),
              AllOf(Scheme("urn"), Authority(""), Body("isbn:0451450523")));
  EXPECT_THAT(
      parseOrDie("file:///c:/windows/system32/"),
      AllOf(Scheme("file"), Authority(""), Body("/c:/windows/system32/")));
}

TEST(URITest, ParseFailed) {
  auto FailedParse = [](llvm::StringRef U) {
    auto URI = URI::parse(U);
    if (!URI) {
      llvm::consumeError(URI.takeError());
      return true;
    }
    return false;
  };

  // Expect ':' in URI.
  EXPECT_TRUE(FailedParse("file//x/y/z"));
  // Empty.
  EXPECT_TRUE(FailedParse(""));
  EXPECT_TRUE(FailedParse(":/a/b/c"));
  EXPECT_TRUE(FailedParse("\"/a/b/c\" IWYU pragma: abc"));
}

TEST(URITest, Resolve) {
#ifdef _WIN32
  EXPECT_THAT(resolveOrDie(parseOrDie("file:///c%3a/x/y/z")), "c:\\x\\y\\z");
#else
  EXPECT_EQ(resolveOrDie(parseOrDie("file:/a/b/c")), "/a/b/c");
  EXPECT_EQ(resolveOrDie(parseOrDie("file://auth/a/b/c")), "/a/b/c");
  EXPECT_THAT(resolveOrDie(parseOrDie("file://au%3dth/%28x%29/y/%20z")),
              "/(x)/y/ z");
  EXPECT_THAT(resolveOrDie(parseOrDie("file:///c:/x/y/z")), "c:/x/y/z");
#endif
  EXPECT_EQ(resolveOrDie(parseOrDie("unittest:///a"), testPath("x")),
            testPath("a"));
}

TEST(URITest, Platform) {
  auto Path = testPath("x");
  auto U = URI::create(Path, "file");
  EXPECT_TRUE(static_cast<bool>(U));
  EXPECT_THAT(resolveOrDie(*U), Path);
}

TEST(URITest, ResolveFailed) {
  auto FailedResolve = [](llvm::StringRef Uri) {
    auto Path = URI::resolve(parseOrDie(Uri));
    if (!Path) {
      llvm::consumeError(Path.takeError());
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
