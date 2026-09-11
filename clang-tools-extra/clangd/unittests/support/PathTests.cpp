//===-- PathTests.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestFS.h"
#include "support/Path.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {
TEST(PathTests, IsAncestor) {
  EXPECT_TRUE(pathStartsWith(testPath("foo"), testPath("foo")));
  EXPECT_TRUE(pathStartsWith(testPath("foo/"), testPath("foo")));

  EXPECT_FALSE(pathStartsWith(testPath("foo"), testPath("fooz")));
  EXPECT_FALSE(pathStartsWith(testPath("foo/"), testPath("fooz")));

  EXPECT_TRUE(pathStartsWith(testPath("foo"), testPath("foo/bar")));
  EXPECT_TRUE(pathStartsWith(testPath("foo/"), testPath("foo/bar")));

#ifdef CLANGD_PATH_CASE_INSENSITIVE
  EXPECT_TRUE(pathStartsWith(testPath("fOo"), testPath("foo/bar")));
  EXPECT_TRUE(pathStartsWith(testPath("foo"), testPath("fOo/bar")));
#else
  EXPECT_FALSE(pathStartsWith(testPath("fOo"), testPath("foo/bar")));
  EXPECT_FALSE(pathStartsWith(testPath("foo"), testPath("fOo/bar")));
#endif
}

TEST(PathTests, MapPathAfterRenames) {
  EXPECT_THAT_EXPECTED(
      mapPathAfterRenames(testPath("old/nested/file.cc"),
                          {{testPath("old"), testPath("new")}}),
      llvm::HasValue(testPath("new/nested/file.cc")));
  EXPECT_THAT_EXPECTED(
      mapPathAfterRenames("relative.cc", {{testPath("old"), testPath("new")}}),
      llvm::HasValue("relative.cc"));
  EXPECT_THAT_EXPECTED(
      mapPathAfterRenames(testPath("old/file.cc"),
                          {{testPath("old/"), testPath("new/./")}}),
      llvm::HasValue(testPath("new/file.cc")));
  EXPECT_THAT_EXPECTED(
      mapPathAfterRenames(testPath("a/file.cc"),
                          {{testPath("a"), testPath("b")},
                           {testPath("a/file.cc"), testPath("c.cc")}}),
      llvm::Failed());
}
} // namespace
} // namespace clangd
} // namespace clang
