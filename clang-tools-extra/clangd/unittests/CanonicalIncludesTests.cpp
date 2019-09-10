//===-- CanonicalIncludesTests.cpp - --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "index/CanonicalIncludes.h"
#include "clang/Basic/LangOptions.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

TEST(CanonicalIncludesTest, CStandardLibrary) {
  CanonicalIncludes CI;
  auto Language = LangOptions();
  Language.C11 = true;
  CI.addSystemHeadersMapping(Language);
  // Usual standard library symbols are mapped correctly.
  EXPECT_EQ("<stdio.h>", CI.mapHeader("path/stdio.h", "printf"));
}

TEST(CanonicalIncludesTest, CXXStandardLibrary) {
  CanonicalIncludes CI;
  auto Language = LangOptions();
  Language.CPlusPlus = true;
  CI.addSystemHeadersMapping(Language);

  // Usual standard library symbols are mapped correctly.
  EXPECT_EQ("<vector>", CI.mapHeader("path/vector.h", "std::vector"));
  EXPECT_EQ("<cstdio>", CI.mapHeader("path/stdio.h", "std::printf"));
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
  LangOptions Language;
  Language.CPlusPlus = true;
  // Ensures 'std::vector' is mapped to '<vector>'.
  CI.addSystemHeadersMapping(Language);

  EXPECT_EQ("<vector>", CI.mapHeader("foo/bar", "std::vector"));
  EXPECT_EQ("foo/bar", CI.mapHeader("foo/bar", "other::symbol"));
}

TEST(CanonicalIncludesTest, Precedence) {
  CanonicalIncludes CI;
  CI.addMapping("some/path", "<path>");
  LangOptions Language;
  Language.CPlusPlus = true;
  CI.addSystemHeadersMapping(Language);

  // We added a mapping from some/path to <path>.
  ASSERT_EQ("<path>", CI.mapHeader("some/path", ""));
  // We should have a path from 'bits/stl_vector.h' to '<vector>'.
  ASSERT_EQ("<vector>", CI.mapHeader("bits/stl_vector.h", ""));
  // We should also have a symbol mapping from 'std::map' to '<map>'.
  ASSERT_EQ("<map>", CI.mapHeader("some/header.h", "std::map"));

  // And the symbol mapping should take precedence over paths mapping.
  EXPECT_EQ("<map>", CI.mapHeader("bits/stl_vector.h", "std::map"));
  EXPECT_EQ("<map>", CI.mapHeader("some/path", "std::map"));
}

} // namespace
} // namespace clangd
} // namespace clang
