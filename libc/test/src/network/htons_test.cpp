//===-- Unittests for htons -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/endian.h"
#include "src/network/htons.h"
#include "src/network/ntohs.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcHtons, SmokeTest) {
  uint16_t original = 0x2301;
  uint16_t swapped = 0x0123;
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  EXPECT_EQ(LIBC_NAMESPACE::htons(original), swapped);
#endif
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  EXPECT_EQ(LIBC_NAMESPACE::htons(original), original);
#endif
}

TEST(LlvmLibcHtons, CompleteTest) {
  uint16_t original = 0x0123;
  EXPECT_EQ(LIBC_NAMESPACE::htons(LIBC_NAMESPACE::ntohs(original)), original);
}
