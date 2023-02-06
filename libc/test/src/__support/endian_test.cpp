//===-- Unittests for endian ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/endian.h"
#include "test/UnitTest/Test.h"

namespace __llvm_libc {

struct LlvmLibcEndian : testing::Test {
  template <typename T> void check(const T original, const T swapped) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    EXPECT_EQ(Endian::to_little_endian(original), original);
    EXPECT_EQ(Endian::to_big_endian(original), swapped);
#endif
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    EXPECT_EQ(Endian::to_big_endian(original), original);
    EXPECT_EQ(Endian::to_little_endian(original), swapped);
#endif
  }
};

TEST_F(LlvmLibcEndian, Field) {
  EXPECT_EQ(Endian::IS_LITTLE, __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__);
  EXPECT_EQ(Endian::IS_BIG, __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__);
}

TEST_F(LlvmLibcEndian, uint8_t) {
  const uint8_t original = uint8_t(0x12);
  check(original, original);
}

TEST_F(LlvmLibcEndian, uint16_t) {
  const uint16_t original = uint16_t(0x1234);
  const uint16_t swapped = __builtin_bswap16(original);
  check(original, swapped);
}

TEST_F(LlvmLibcEndian, uint32_t) {
  const uint32_t original = uint32_t(0x12345678);
  const uint32_t swapped = __builtin_bswap32(original);
  check(original, swapped);
}

TEST_F(LlvmLibcEndian, uint64_t) {
  const uint64_t original = uint64_t(0x123456789ABCDEF0);
  const uint64_t swapped = __builtin_bswap64(original);
  check(original, swapped);
}

} // namespace __llvm_libc
