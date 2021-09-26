//===-- extensible_address_test.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of the ORC runtime.
//
// Note:
//   This unit test was adapted from
//   llvm/unittests/Support/ExecutorAddressTest.cpp
//
//===----------------------------------------------------------------------===//

#include "executor_address.h"
#include "gtest/gtest.h"

using namespace __orc_rt;

TEST(ExecutorAddrTest, DefaultAndNull) {
  // Check that default constructed values and isNull behave as expected.

  ExecutorAddr Default;
  ExecutorAddr Null(0);
  ExecutorAddr NonNull(1);

  EXPECT_TRUE(Null.isNull());
  EXPECT_EQ(Default, Null);

  EXPECT_FALSE(NonNull.isNull());
  EXPECT_NE(Default, NonNull);
}

TEST(ExecutorAddrTest, Ordering) {
  // Check that ordering operations.
  ExecutorAddr A1(1), A2(2);

  EXPECT_LE(A1, A1);
  EXPECT_LT(A1, A2);
  EXPECT_GT(A2, A1);
  EXPECT_GE(A2, A2);
}

TEST(ExecutorAddrTest, PtrConversion) {
  // Test toPtr / fromPtr round-tripping.
  int X = 0;
  auto XAddr = ExecutorAddr::fromPtr(&X);
  int *XPtr = XAddr.toPtr<int *>();

  EXPECT_EQ(XPtr, &X);
}

TEST(ExecutorAddrTest, AddrRanges) {
  ExecutorAddr A0(0), A1(1), A2(2), A3(3);
  ExecutorAddrRange R0(A0, A1), R1(A1, A2), R2(A2, A3), R3(A0, A2), R4(A1, A3);
  //     012
  // R0: #      -- Before R1
  // R1:  #     --
  // R2:   #    -- After R1
  // R3: ##     -- Overlaps R1 start
  // R4:  ##    -- Overlaps R1 end

  EXPECT_EQ(R1, ExecutorAddrRange(A1, A2));
  EXPECT_EQ(R1, ExecutorAddrRange(A1, ExecutorAddrDiff(1)));
  EXPECT_NE(R1, R2);

  EXPECT_TRUE(R1.contains(A1));
  EXPECT_FALSE(R1.contains(A0));
  EXPECT_FALSE(R1.contains(A2));

  EXPECT_FALSE(R1.overlaps(R0));
  EXPECT_FALSE(R1.overlaps(R2));
  EXPECT_TRUE(R1.overlaps(R3));
  EXPECT_TRUE(R1.overlaps(R4));
}
