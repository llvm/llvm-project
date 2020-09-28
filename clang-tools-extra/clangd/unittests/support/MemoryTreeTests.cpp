//===-- MemoryTreeTests.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/MemoryTree.h"
#include "llvm/Support/Allocator.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <ostream>

namespace clang {
namespace clangd {
namespace {
using testing::Contains;
using testing::IsEmpty;
using testing::UnorderedElementsAre;

MATCHER_P2(WithNameAndSize, Name, Size, "") {
  return arg.first == Name &&
         arg.getSecond().total() == static_cast<size_t>(Size);
}

TEST(MemoryTree, Basics) {
  MemoryTree MT;
  EXPECT_EQ(MT.total(), 0U);
  EXPECT_THAT(MT.children(), IsEmpty());

  MT.addUsage(42);
  EXPECT_EQ(MT.total(), 42U);
  EXPECT_THAT(MT.children(), IsEmpty());

  MT.child("leaf").addUsage(1);
  EXPECT_EQ(MT.total(), 43U);
  EXPECT_THAT(MT.children(), UnorderedElementsAre(WithNameAndSize("leaf", 1)));

  // child should be idempotent.
  MT.child("leaf").addUsage(1);
  EXPECT_EQ(MT.total(), 44U);
  EXPECT_THAT(MT.children(), UnorderedElementsAre(WithNameAndSize("leaf", 2)));
}

TEST(MemoryTree, DetailedNodesWithoutDetails) {
  MemoryTree MT;
  MT.detail("should_be_ignored").addUsage(2);
  EXPECT_THAT(MT.children(), IsEmpty());
  EXPECT_EQ(MT.total(), 2U);

  // Make sure children from details are merged.
  MT.detail("first_detail").child("leaf").addUsage(1);
  MT.detail("second_detail").child("leaf").addUsage(1);
  EXPECT_THAT(MT.children(), Contains(WithNameAndSize("leaf", 2)));
}

TEST(MemoryTree, DetailedNodesWithDetails) {
  llvm::BumpPtrAllocator Alloc;
  MemoryTree MT(&Alloc);

  {
    auto &Detail = MT.detail("first_detail");
    Detail.child("leaf").addUsage(1);
    EXPECT_THAT(MT.children(), Contains(WithNameAndSize("first_detail", 1)));
    EXPECT_THAT(Detail.children(), Contains(WithNameAndSize("leaf", 1)));
  }

  {
    auto &Detail = MT.detail("second_detail");
    Detail.child("leaf").addUsage(1);
    EXPECT_THAT(MT.children(), Contains(WithNameAndSize("second_detail", 1)));
    EXPECT_THAT(Detail.children(), Contains(WithNameAndSize("leaf", 1)));
  }
}
} // namespace
} // namespace clangd
} // namespace clang
