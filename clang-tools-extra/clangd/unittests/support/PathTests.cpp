//===-- PathTests.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestFS.h"
#include "support/Path.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {
TEST(PathTests, IsAncestor) {
  EXPECT_TRUE(PathRef(testPath("foo")).startsWith(testPath("foo")));
  EXPECT_TRUE(PathRef(testPath("foo/")).startsWith(testPath("foo")));

  EXPECT_FALSE(PathRef(testPath("foo")).startsWith(testPath("fooz")));
  EXPECT_FALSE(PathRef(testPath("foo/")).startsWith(testPath("fooz")));

  EXPECT_TRUE(PathRef(testPath("foo")).startsWith(testPath("foo/bar")));
  EXPECT_TRUE(PathRef(testPath("foo/")).startsWith(testPath("foo/bar")));

#ifdef CLANGD_PATH_CASE_INSENSITIVE
  EXPECT_TRUE(PathRef(testPath("fOo")).startsWith(testPath("foo/bar")));
  EXPECT_TRUE(PathRef(testPath("foo")).startsWith(testPath("fOo/bar")));
#else
  EXPECT_FALSE(PathRef(testPath("fOo")).startsWith(testPath("foo/bar")));
  EXPECT_FALSE(PathRef(testPath("foo")).startsWith(testPath("fOo/bar")));
#endif
}
} // namespace
} // namespace clangd
} // namespace clang
