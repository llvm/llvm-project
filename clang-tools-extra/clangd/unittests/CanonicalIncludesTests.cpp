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
  EXPECT_EQ("<stdio.h>", CI.mapSymbol("printf"));
  EXPECT_EQ("", CI.mapSymbol("unknown_symbol"));
}

TEST(CanonicalIncludesTest, CXXStandardLibrary) {
  CanonicalIncludes CI;
  auto Language = LangOptions();
  Language.CPlusPlus = true;
  CI.addSystemHeadersMapping(Language);

  // Usual standard library symbols are mapped correctly.
  EXPECT_EQ("<vector>", CI.mapSymbol("std::vector"));
  EXPECT_EQ("<cstdio>", CI.mapSymbol("std::printf"));
  // std::move is ambiguous, currently always mapped to <utility>
  EXPECT_EQ("<utility>", CI.mapSymbol("std::move"));
  // Unknown std symbols aren't mapped.
  EXPECT_EQ("", CI.mapSymbol("std::notathing"));
  // iosfwd declares some symbols it doesn't own.
  EXPECT_EQ("<ostream>", CI.mapSymbol("std::ostream"));
  // And (for now) we assume it owns the others.
  EXPECT_EQ("<iosfwd>", CI.mapHeader("iosfwd"));
}

TEST(CanonicalIncludesTest, PathMapping) {
  // As used for IWYU pragmas.
  CanonicalIncludes CI;
  CI.addMapping("foo/bar", "<baz>");

  EXPECT_EQ("<baz>", CI.mapHeader("foo/bar"));
  EXPECT_EQ("", CI.mapHeader("bar/bar"));
}

TEST(CanonicalIncludesTest, Precedence) {
  CanonicalIncludes CI;
  CI.addMapping("some/path", "<path>");
  LangOptions Language;
  Language.CPlusPlus = true;
  CI.addSystemHeadersMapping(Language);

  // We added a mapping from some/path to <path>.
  ASSERT_EQ("<path>", CI.mapHeader("some/path"));
  // We should have a path from 'bits/stl_vector.h' to '<vector>'.
  ASSERT_EQ("<vector>", CI.mapHeader("bits/stl_vector.h"));
}

} // namespace
} // namespace clangd
} // namespace clang
