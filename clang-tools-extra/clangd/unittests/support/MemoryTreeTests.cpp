//===-- MemoryTreeTests.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/MemoryTree.h"
#include "support/TestTracer.h"
#include "support/Trace.h"
#include "llvm/Support/Allocator.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <ostream>

namespace clang {
namespace clangd {
namespace {
using testing::Contains;
using testing::ElementsAre;
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

TEST(MemoryTree, Record) {
  trace::TestTracer Tracer;
  static constexpr llvm::StringLiteral MetricName = "memory_usage";
  static constexpr trace::Metric OutMetric(MetricName, trace::Metric::Value,
                                           "component_name");
  auto AddNodes = [](MemoryTree Root) {
    Root.child("leaf").addUsage(1);

    {
      auto &Detail = Root.detail("detail");
      Detail.addUsage(1);
      Detail.child("leaf").addUsage(1);
      auto &Child = Detail.child("child");
      Child.addUsage(1);
      Child.child("leaf").addUsage(1);
    }

    {
      auto &Child = Root.child("child");
      Child.addUsage(1);
      Child.child("leaf").addUsage(1);
    }
    return Root;
  };

  llvm::BumpPtrAllocator Alloc;
  record(AddNodes(MemoryTree(&Alloc)), "root", OutMetric);
  EXPECT_THAT(Tracer.takeMetric(MetricName, "root"), ElementsAre(7));
  EXPECT_THAT(Tracer.takeMetric(MetricName, "root.leaf"), ElementsAre(1));
  EXPECT_THAT(Tracer.takeMetric(MetricName, "root.detail"), ElementsAre(4));
  EXPECT_THAT(Tracer.takeMetric(MetricName, "root.detail.leaf"),
              ElementsAre(1));
  EXPECT_THAT(Tracer.takeMetric(MetricName, "root.detail.child"),
              ElementsAre(2));
  EXPECT_THAT(Tracer.takeMetric(MetricName, "root.detail.child.leaf"),
              ElementsAre(1));
  EXPECT_THAT(Tracer.takeMetric(MetricName, "root.child"), ElementsAre(2));
  EXPECT_THAT(Tracer.takeMetric(MetricName, "root.child.leaf"), ElementsAre(1));
}
} // namespace
} // namespace clangd
} // namespace clang
