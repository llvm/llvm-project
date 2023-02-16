//===-- Unittests for htonl -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/endian.h"
#include "src/network/htonl.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcHtonl, SmokeTest) {
  uint32_t original = 0x67452301;
  uint32_t swapped = 0x01234567;
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  EXPECT_EQ(__llvm_libc::htonl(original), swapped);
#endif
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  EXPECT_EQ(__llvm_libc::htonl(original), original);
#endif
}
