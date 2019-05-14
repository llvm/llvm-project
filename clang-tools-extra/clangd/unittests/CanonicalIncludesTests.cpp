//===-- CanonicalIncludesTests.cpp - --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "index/CanonicalIncludes.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

TEST(CanonicalIncludesTest, CXXStandardLibrary) {
  CanonicalIncludes CI;
  addSystemHeadersMapping(&CI);

  // Usual standard library symbols are mapped correctly.
  EXPECT_EQ("<vector>", CI.mapHeader("path/vector.h", "std::vector"));
  // std::move is ambiguous, currently mapped only based on path
  EXPECT_EQ("<utility>", CI.mapHeader("libstdc++/bits/move.h", "std::move"));
  EXPECT_EQ("path/utility.h", CI.mapHeader("path/utility.h", "std::move"));
  // Unknown std symbols aren't mapped.
  EXPECT_EQ("foo/bar.h", CI.mapHeader("foo/bar.h", "std::notathing"));
  // iosfwd declares some symbols it doesn't own.
  EXPECT_EQ("<ostream>", CI.mapHeader("iosfwd", "std::ostream"));
  // And (for now) we assume it owns the others.
  EXPECT_EQ("<iosfwd>", CI.mapHeader("iosfwd", "std::notwathing"));
}

TEST(CanonicalIncludesTest, PathMapping) {
  // As used for IWYU pragmas.
  CanonicalIncludes CI;
  CI.addMapping("foo/bar", "<baz>");

  EXPECT_EQ("<baz>", CI.mapHeader("foo/bar", "some::symbol"));
  EXPECT_EQ("bar/bar", CI.mapHeader("bar/bar", "some::symbol"));
}

TEST(CanonicalIncludesTest, SymbolMapping) {
  // As used for standard library.
  CanonicalIncludes CI;
  CI.addSymbolMapping("some::symbol", "<baz>");

  EXPECT_EQ("<baz>", CI.mapHeader("foo/bar", "some::symbol"));
  EXPECT_EQ("foo/bar", CI.mapHeader("foo/bar", "other::symbol"));
}

TEST(CanonicalIncludesTest, Precedence) {
  CanonicalIncludes CI;
  CI.addMapping("some/path", "<path>");
  CI.addSymbolMapping("some::symbol", "<symbol>");

  // Symbol mapping beats path mapping.
  EXPECT_EQ("<symbol>", CI.mapHeader("some/path", "some::symbol"));
}

} // namespace
} // namespace clangd
} // namespace clang
