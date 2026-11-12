//===-- Unittests for bitmask ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/HashTable/bitmask.h"
#include "src/__support/macros/config.h"
#include "test/UnitTest/Test.h"
namespace LIBC_NAMESPACE_DECL {
namespace internal {

using ShortBitMask = BitMaskAdaptor<uint16_t, 1>;
using LargeBitMask = BitMaskAdaptor<uint64_t, 8>;

TEST(LlvmLibcHashTableBitMaskTest, SingleBitStrideLowestSetBit) {
  uint16_t data = 0xffff;
  for (size_t i = 0; i < 16; ++i) {
    if (ShortBitMask{data}.any_bit_set()) {
      ASSERT_EQ(ShortBitMask{data}.lowest_set_bit_nonzero(), i);
      data <<= 1;
    }
  }
}

TEST(LlvmLibcHashTableBitMaskTest, MultiBitStrideLowestSetBit) {
  uint64_t data = 0xffff'ffff'ffff'ffff;
  for (size_t i = 0; i < 8; ++i) {
    for (size_t j = 0; j < 8; ++j) {
      if (LargeBitMask{data}.any_bit_set()) {
        ASSERT_EQ(LargeBitMask{data}.lowest_set_bit_nonzero(), i);
        data <<= 1;
      }
    }
  }
}

TEST(LlvmLibcHashTableBitMaskTest, SingleBitStrideIteration) {
  using Iter = IteratableBitMaskAdaptor<ShortBitMask>;
  uint16_t data = 0xffff;
  for (size_t i = 0; i < 16; ++i) {
    Iter iter = {data};
    size_t j = i;
    for (auto x : iter) {
      ASSERT_EQ(x, j);
      j++;
    }
    ASSERT_EQ(j, size_t{16});
    data <<= 1;
  }
}

TEST(LlvmLibcHashTableBitMaskTest, MultiBitStrideIteration) {
  using Iter = IteratableBitMaskAdaptor<LargeBitMask>;
  uint64_t data = 0x8080808080808080ul;
  for (size_t i = 0; i < 8; ++i) {
    Iter iter = {data};
    size_t j = i;
    for (auto x : iter) {
      ASSERT_EQ(x, j);
      j++;
    }
    ASSERT_EQ(j, size_t{8});
    data <<= Iter::STRIDE;
  }
}
} // namespace internal
} // namespace LIBC_NAMESPACE_DECL
