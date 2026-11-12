//===- unittests/Basic/LineOffsetMappingTest.cpp - Test LineOffsetMapping -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/SourceManager.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace clang::SrcMgr;
using namespace llvm;

namespace {

TEST(LineOffsetMappingTest, empty) {
  LineOffsetMapping Mapping;
  EXPECT_FALSE(Mapping);

#if !defined(NDEBUG) && GTEST_HAS_DEATH_TEST
  EXPECT_DEATH((void)Mapping.getLines(), "Storage");
#endif
}

TEST(LineOffsetMappingTest, construct) {
  BumpPtrAllocator Alloc;
  unsigned Offsets[] = {0, 10, 20};
  LineOffsetMapping Mapping(Offsets, Alloc);
  EXPECT_EQ(3u, Mapping.size());
  EXPECT_EQ(0u, Mapping[0]);
  EXPECT_EQ(10u, Mapping[1]);
  EXPECT_EQ(20u, Mapping[2]);

#if !defined(NDEBUG) && GTEST_HAS_DEATH_TEST
  EXPECT_DEATH((void)Mapping[3], "Invalid index");
#endif
}

TEST(LineOffsetMappingTest, constructTwo) {
  // Confirm allocation size is big enough, convering an off-by-one bug.
  BumpPtrAllocator Alloc;
  unsigned Offsets1[] = {0, 10};
  unsigned Offsets2[] = {0, 20};
  LineOffsetMapping Mapping1(Offsets1, Alloc);
  LineOffsetMapping Mapping2(Offsets2, Alloc);

  // Need to check Mapping1 *after* building Mapping2.
  EXPECT_EQ(2u, Mapping1.size());
  EXPECT_EQ(0u, Mapping1[0]);
  EXPECT_EQ(10u, Mapping1[1]);
  EXPECT_EQ(2u, Mapping2.size());
  EXPECT_EQ(0u, Mapping2[0]);
  EXPECT_EQ(20u, Mapping2[1]);
}

TEST(LineOffsetMappingTest, get) {
  BumpPtrAllocator Alloc;
  StringRef Source = "first line\n"
                     "second line\n";
  auto Mapping = LineOffsetMapping::get(MemoryBufferRef(Source, ""), Alloc);
  EXPECT_EQ(3u, Mapping.size());
  EXPECT_EQ(0u, Mapping[0]);
  EXPECT_EQ(11u, Mapping[1]);
  EXPECT_EQ(23u, Mapping[2]);
}

TEST(LineOffsetMappingTest, getMissingFinalNewline) {
  BumpPtrAllocator Alloc;
  StringRef Source = "first line\n"
                     "second line";
  auto Mapping = LineOffsetMapping::get(MemoryBufferRef(Source, ""), Alloc);
  EXPECT_EQ(2u, Mapping.size());
  EXPECT_EQ(0u, Mapping[0]);
  EXPECT_EQ(11u, Mapping[1]);
}

} // end namespace
