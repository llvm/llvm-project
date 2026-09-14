//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for InlineAny.
///
//===----------------------------------------------------------------------===//

#include "src/__support/inline_any.h"
#include "src/__support/uint128.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

struct PageAlignedInt {
  alignas(64) int value;
};

TEST(LlvmLibcInlineAnyTest, LoadAndStore) {
  LIBC_NAMESPACE::InlineAny<sizeof(UInt128), alignof(UInt128)> any;

  any.store(27);
  EXPECT_EQ(any.load<int>(), 27);

  any.store(-5.25);
  EXPECT_FP_EQ(any.load<double>(), -5.25);

  any.store(UInt128(1) << 75);
  EXPECT_EQ(any.load<UInt128>(), UInt128(1) << 75);

  LIBC_NAMESPACE::InlineAny<sizeof(PageAlignedInt), alignof(PageAlignedInt)>
      overaligned_any;

  overaligned_any.store(42);
  EXPECT_EQ(overaligned_any.load<int>(), 42);

  overaligned_any.store(PageAlignedInt{-17});
  EXPECT_EQ(overaligned_any.load<PageAlignedInt>().value, -17);
}
