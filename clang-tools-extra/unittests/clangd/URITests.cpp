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
namespace {

using ::testing::AllOf;

MATCHER_P(Scheme, S, "") { return arg.scheme() == S; }
MATCHER_P(Authority, A, "") { return arg.authority() == A; }
MATCHER_P(Body, B, "") { return arg.body() == B; }

// Assume all files in the schema have a "test-root/" root directory, and the
// schema path is the relative path to the root directory.
// So the schema of "/some-dir/test-root/x/y/z" is "test:x/y/z".
class TestScheme : public URIScheme {
public:
  static const char *Scheme;

  static const char *TestRoot;

  llvm::Expected<std::string>
  getAbsolutePath(llvm::StringRef /*Authority*/, llvm::StringRef Body,
                  llvm::StringRef HintPath) const override {
    auto Pos = HintPath.find(TestRoot);
    assert(Pos != llvm::StringRef::npos);
    return (HintPath.substr(0, Pos + llvm::StringRef(TestRoot).size()) + Body)
        .str();
  }

  llvm::Expected<FileURI>
  uriFromAbsolutePath(llvm::StringRef AbsolutePath) const override {
    auto Pos = AbsolutePath.find(TestRoot);
    assert(Pos != llvm::StringRef::npos);
    return FileURI::create(
        Scheme, /*Authority=*/"",
        AbsolutePath.substr(Pos + llvm::StringRef(TestRoot).size()));
  }
};

const char *TestScheme::Scheme = "test";
const char *TestScheme::TestRoot = "/test-root/";

static URISchemeRegistry::Add<TestScheme> X(TestScheme::Scheme, "Test schema");

std::string createOrDie(llvm::StringRef AbsolutePath,
                        llvm::StringRef Scheme = "file") {
  auto Uri = FileURI::create(AbsolutePath, Scheme);
  if (!Uri)
    llvm_unreachable(llvm::toString(Uri.takeError()).c_str());
  return Uri->toString();
}

std::string createOrDie(llvm::StringRef Scheme, llvm::StringRef Authority,
                        llvm::StringRef Body) {
  auto Uri = FileURI::create(Scheme, Authority, Body);
  if (!Uri)
    llvm_unreachable(llvm::toString(Uri.takeError()).c_str());
  return Uri->toString();
}

FileURI parseOrDie(llvm::StringRef Uri) {
  auto U = FileURI::parse(Uri);
  if (!U)
    llvm_unreachable(llvm::toString(U.takeError()).c_str());
  return *U;
}

TEST(PercentEncodingTest, Encode) {
  EXPECT_EQ(createOrDie("x", /*Authority=*/"", "a/b/c"), "x:a/b/c");
  EXPECT_EQ(createOrDie("x", /*Authority=*/"", "a!b;c~"), "x:a%21b%3bc~");
}

TEST(PercentEncodingTest, Decode) {
  EXPECT_EQ(parseOrDie("x:a/b/c").body(), "a/b/c");

  EXPECT_EQ(parseOrDie("%3a://%3a/%3").scheme(), ":");
  EXPECT_EQ(parseOrDie("%3a://%3a/%3").authority(), ":");
  EXPECT_EQ(parseOrDie("%3a://%3a/%3").body(), "/%3");

  EXPECT_EQ(parseOrDie("x:a%21b%3ac~").body(), "a!b:c~");
}

std::string resolveOrDie(const FileURI &U, llvm::StringRef HintPath = "") {
  auto Path = FileURI::resolve(U, HintPath);
  if (!Path)
    llvm_unreachable(llvm::toString(Path.takeError()).c_str());
  return *Path;
}

TEST(URITest, Create) {
#ifdef LLVM_ON_WIN32
  EXPECT_THAT(createOrDie("c:\\x\\y\\z"), "file:///c%3a/x/y/z");
#else
  EXPECT_THAT(createOrDie("/x/y/z"), "file:///x/y/z");
  EXPECT_THAT(createOrDie("/(x)/y/\\ z"), "file:///%28x%29/y/%5c%20z");
#endif
}

TEST(URITest, FailedCreate) {
  auto Fail = [](llvm::Expected<FileURI> U) {
    if (!U) {
      llvm::consumeError(U.takeError());
      return true;
    }
    return false;
  };
  // Create from scheme+authority+body:
  //
  // Scheme must be provided.
  EXPECT_TRUE(Fail(FileURI::create("", "auth", "/a")));
  // Body must start with '/' if authority is present.
  EXPECT_TRUE(Fail(FileURI::create("scheme", "auth", "x/y/z")));

  // Create from scheme registry:
  //
  EXPECT_TRUE(Fail(FileURI::create("/x/y/z", "no")));
  // Path has to be absolute.
  EXPECT_TRUE(Fail(FileURI::create("x/y/z")));
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
    auto URI = FileURI::parse(U);
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
}

TEST(URITest, Resolve) {
#ifdef LLVM_ON_WIN32
  EXPECT_THAT(resolveOrDie(parseOrDie("file:///c%3a/x/y/z")), "c:\\x\\y\\z");
#else
  EXPECT_EQ(resolveOrDie(parseOrDie("file:/a/b/c")), "/a/b/c");
  EXPECT_EQ(resolveOrDie(parseOrDie("file://auth/a/b/c")), "/a/b/c");
  EXPECT_EQ(resolveOrDie(parseOrDie("test:a/b/c"), "/dir/test-root/x/y/z"),
            "/dir/test-root/a/b/c");
  EXPECT_THAT(resolveOrDie(parseOrDie("file://au%3dth/%28x%29/y/%20z")),
              "/(x)/y/ z");
  EXPECT_THAT(resolveOrDie(parseOrDie("file:///c:/x/y/z")), "c:/x/y/z");
#endif
}

TEST(URITest, Platform) {
  auto Path = getVirtualTestFilePath("x");
  auto U = FileURI::create(Path, "file");
  EXPECT_TRUE(static_cast<bool>(U));
  EXPECT_THAT(resolveOrDie(*U), Path.str());
}

TEST(URITest, ResolveFailed) {
  auto FailedResolve = [](llvm::StringRef Uri) {
    auto Path = FileURI::resolve(parseOrDie(Uri));
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
