//===- unittests/Analysis/FlowSensitive/SourceLocationsLatticeTest.cpp ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/SourceLocationsLattice.h"

#include "clang/Analysis/FlowSensitive/DataflowLattice.h"
#include "clang/Basic/SourceLocation.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace dataflow {
namespace {

TEST(SourceLocationsLatticeTest, Comparison) {
  const SourceLocationsLattice Bottom;
  const SourceLocationsLattice NonBottom(
      {SourceLocation::getFromRawEncoding(0)});

  EXPECT_TRUE(Bottom == Bottom);
  EXPECT_FALSE(Bottom == NonBottom);
  EXPECT_FALSE(NonBottom == Bottom);
  EXPECT_TRUE(NonBottom == NonBottom);

  EXPECT_FALSE(Bottom != Bottom);
  EXPECT_TRUE(Bottom != NonBottom);
  EXPECT_TRUE(NonBottom != Bottom);
  EXPECT_FALSE(NonBottom != NonBottom);
}

TEST(SourceLocationsLatticeTest, Join) {
  const SourceLocationsLattice Bottom;
  const SourceLocationsLattice NonBottom(
      {SourceLocation::getFromRawEncoding(0)});
  {
    SourceLocationsLattice LHS = Bottom;
    const SourceLocationsLattice RHS = Bottom;
    EXPECT_EQ(LHS.join(RHS), LatticeJoinEffect::Unchanged);
    EXPECT_EQ(LHS, Bottom);
  }
  {
    SourceLocationsLattice LHS = NonBottom;
    const SourceLocationsLattice RHS = Bottom;
    EXPECT_EQ(LHS.join(RHS), LatticeJoinEffect::Unchanged);
    EXPECT_EQ(LHS, NonBottom);
  }
  {
    SourceLocationsLattice LHS = Bottom;
    const SourceLocationsLattice RHS = NonBottom;
    EXPECT_EQ(LHS.join(RHS), LatticeJoinEffect::Changed);
    EXPECT_EQ(LHS, NonBottom);
  }
  {
    SourceLocationsLattice LHS = NonBottom;
    const SourceLocationsLattice RHS = NonBottom;
    EXPECT_EQ(LHS.join(RHS), LatticeJoinEffect::Unchanged);
    EXPECT_EQ(LHS, NonBottom);
  }
}

} // namespace
} // namespace dataflow
} // namespace clang
