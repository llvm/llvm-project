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
#include <chrono>
#include <optional>
#include <utility>

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

  std::pair<std::string, /*Parsed=*/bool>
  get(std::chrono::steady_clock::time_point FreshTime) const {
    bool GotParse = false;
    bool GotRead = false;
    std::string Result;
    read(
        FS, FreshTime,
        [&](std::optional<llvm::StringRef> Data) {
          GotParse = true;
          Value = Data.value_or("").str();
        },
        [&]() {
          GotRead = true;
          Result = Value;
        });
    EXPECT_TRUE(GotRead);
    return {Result, GotParse};
  }
};

MATCHER_P(Parsed, Value, "") { return arg.second && arg.first == Value; }
MATCHER_P(Cached, Value, "") { return !arg.second && arg.first == Value; }

TEST(FileCacheTest, Invalidation) {
  TestCache C;

  auto StaleOK = std::chrono::steady_clock::now();
  auto MustBeFresh = StaleOK + std::chrono::hours(1);

  C.setContents("a");
  EXPECT_THAT(C.get(StaleOK), Parsed("a")) << "Parsed first time";
  EXPECT_THAT(C.get(StaleOK), Cached("a")) << "Cached (time)";
  EXPECT_THAT(C.get(MustBeFresh), Cached("a")) << "Cached (stat)";
  C.setContents("bb");
  EXPECT_THAT(C.get(StaleOK), Cached("a")) << "Cached (time)";
  EXPECT_THAT(C.get(MustBeFresh), Parsed("bb")) << "Size changed";
  EXPECT_THAT(C.get(MustBeFresh), Cached("bb")) << "Cached (stat)";
  C.setContents(nullptr);
  EXPECT_THAT(C.get(StaleOK), Cached("bb")) << "Cached (time)";
  EXPECT_THAT(C.get(MustBeFresh), Parsed("")) << "stat failed";
  EXPECT_THAT(C.get(MustBeFresh), Cached("")) << "Cached (404)";
  C.setContents("bb"); // Match the previous stat values!
  EXPECT_THAT(C.get(StaleOK), Cached("")) << "Cached (time)";
  EXPECT_THAT(C.get(MustBeFresh), Parsed("bb")) << "Size changed";
}

} // namespace
} // namespace config
} // namespace clangd
} // namespace clang
