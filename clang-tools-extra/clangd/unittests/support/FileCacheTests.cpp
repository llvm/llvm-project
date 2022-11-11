//===-- FileCacheTests.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/FileCache.h"

#include "TestFS.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <atomic>
#include <chrono>

namespace clang {
namespace clangd {
namespace config {
namespace {

class TestCache : public FileCache {
  MockFS FS;
  mutable std::string Value;

public:
  TestCache() : FileCache(testPath("foo.cc")) {}

  void setContents(const char *C) {
    if (C)
      FS.Files[testPath("foo.cc")] = C;
    else
      FS.Files.erase(testPath("foo.cc"));
  }

  std::string get(std::chrono::steady_clock::time_point FreshTime,
                  bool ExpectParse) const {
    bool GotParse = false;
    bool GotRead;
    std::string Result;
    read(
        FS, FreshTime,
        [&](llvm::Optional<llvm::StringRef> Data) {
          GotParse = true;
          Value = Data.value_or("").str();
        },
        [&]() {
          GotRead = true;
          Result = Value;
        });
    EXPECT_EQ(GotParse, ExpectParse);
    EXPECT_TRUE(GotRead);
    return Result;
  }
};

TEST(FileCacheTest, Invalidation) {
  TestCache C;

  auto StaleOK = std::chrono::steady_clock::now();
  auto MustBeFresh = StaleOK + std::chrono::hours(1);

  C.setContents("a");
  EXPECT_EQ("a", C.get(StaleOK, /*ExpectParse=*/true)) << "Parsed first time";
  EXPECT_EQ("a", C.get(StaleOK, /*ExpectParse=*/false)) << "Cached (time)";
  EXPECT_EQ("a", C.get(MustBeFresh, /*ExpectParse=*/false)) << "Cached (stat)";
  C.setContents("bb");
  EXPECT_EQ("a", C.get(StaleOK, /*ExpectParse=*/false)) << "Cached (time)";
  EXPECT_EQ("bb", C.get(MustBeFresh, /*ExpectParse=*/true)) << "Size changed";
  EXPECT_EQ("bb", C.get(MustBeFresh, /*ExpectParse=*/true)) << "Cached (stat)";
  C.setContents(nullptr);
  EXPECT_EQ("bb", C.get(StaleOK, /*ExpectParse=*/false)) << "Cached (time)";
  EXPECT_EQ("", C.get(MustBeFresh, /*ExpectParse=*/true)) << "Stat failed";
  EXPECT_EQ("", C.get(MustBeFresh, /*ExpectParse=*/false)) << "Cached (404)";
  C.setContents("bb"); // Match the previous stat values!
  EXPECT_EQ("", C.get(StaleOK, /*ExpectParse=*/false)) << "Cached (time)";
  EXPECT_EQ("bb", C.get(MustBeFresh, /*ExpectParse=*/true)) << "Size changed";
}

} // namespace
} // namespace config
} // namespace clangd
} // namespace clang
