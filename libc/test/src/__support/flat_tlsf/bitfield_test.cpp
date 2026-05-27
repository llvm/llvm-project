//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for flat_tlsf bit utilities and BitField.
///
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/algorithm.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/flat_tlsf/bit_utils.h"
#include "src/__support/flat_tlsf/bitfield.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::cpp::numeric_limits;
using LIBC_NAMESPACE::flat_tlsf::BitField;
using LIBC_NAMESPACE::flat_tlsf::Byte;
namespace bit_utils = LIBC_NAMESPACE::flat_tlsf::bit_utils;

namespace {

#define EXPECT_BF_EQ(lhs, rhs)                                                 \
  for (size_t _i = 0; _i < BitField::NUMBER_OF_ELEMENTS; ++_i) {               \
    EXPECT_EQ((lhs).storage[_i], (rhs).storage[_i]);                           \
  }

TEST(LlvmLibcFlatTlsfTest, BitUtilsPointerAlignment) {
  alignas(64) Byte bytes[128] = {};
  Byte *ptr = bytes + 37;

  EXPECT_TRUE(bit_utils::is_aligned_to(bytes, 64));
  EXPECT_FALSE(bit_utils::is_aligned_to(ptr, 32));
  EXPECT_EQ(bit_utils::align_down_by(ptr, 32), bytes + 32);
  EXPECT_EQ(bit_utils::align_up_by(ptr, 32), bytes + 64);
  EXPECT_EQ(bit_utils::align_up_by_mask(ptr, 31), bytes + 64);
  EXPECT_EQ(bit_utils::align_up_by(bytes + 64, 32), bytes + 64);
}

TEST(LlvmLibcFlatTlsfTest, BitUtilsSaturatingPtrAdd) {
  Byte *ptr = reinterpret_cast<Byte *>(uintptr_t{100});
  EXPECT_EQ(bit_utils::saturating_ptr_add(ptr, 23),
            reinterpret_cast<Byte *>(uintptr_t{123}));

  Byte *near_end =
      reinterpret_cast<Byte *>(numeric_limits<uintptr_t>::max() - uintptr_t{3});
  EXPECT_EQ(bit_utils::saturating_ptr_add(near_end, 4),
            reinterpret_cast<Byte *>(numeric_limits<uintptr_t>::max()));
}

TEST(LlvmLibcFlatTlsfTest, BitFieldTestBitScanAfter) {
  BitField field = BitField::zeros();
  field.storage[0] = 0b10100;
  field.storage[1] = 0b00010;

  EXPECT_EQ(field.bit_scan_after(0), 2u);
  EXPECT_EQ(field.bit_scan_after(2), 2u);
  EXPECT_EQ(field.bit_scan_after(3), 4u);
  EXPECT_EQ(field.bit_scan_after(4), 4u);
  EXPECT_EQ(field.bit_scan_after(5),
            static_cast<uint32_t>(BitField::BITS_PER_ELEMENT + 1));
  EXPECT_EQ(field.bit_scan_after(BitField::BITS_PER_ELEMENT + 1),
            static_cast<uint32_t>(BitField::BITS_PER_ELEMENT + 1));
  EXPECT_EQ(field.bit_scan_after(BitField::BITS_PER_ELEMENT + 2),
            static_cast<uint32_t>(BitField::BITS));
}

TEST(LlvmLibcFlatTlsfTest, BitFieldSetUnset) {
  for (uint32_t i = 0; i < BitField::BITS; ++i) {
    BitField bf = BitField::zeros();
    bf.set_bit(i);
    bf.clear_bit(i);
    EXPECT_BF_EQ(bf, BitField::zeros());
  }
}

TEST(LlvmLibcFlatTlsfTest, BitFieldSetEqSetAllUnsetRest) {
  for (uint32_t i = 0; i < BitField::BITS; ++i) {
    BitField bf = BitField::zeros();
    for (uint32_t j = 0; j < i; ++j)
      bf.set_bit(j);

    BitField bf2 = BitField::zeros();
    for (uint32_t j = 0; j < BitField::BITS; ++j)
      bf2.set_bit(j);
    for (uint32_t j = i; j < BitField::BITS; ++j)
      bf2.clear_bit(j);

    EXPECT_BF_EQ(bf, bf2);
  }
}

TEST(LlvmLibcFlatTlsfTest, BitFieldBsfZero) {
  BitField bf = BitField::zeros();
  for (uint32_t i = 0; i < BitField::BITS; ++i)
    EXPECT_EQ(bf.bit_scan_after(i), static_cast<uint32_t>(BitField::BITS));
}

TEST(LlvmLibcFlatTlsfTest, BitFieldBsf) {
  for (uint32_t i = 0; i < BitField::BITS; ++i) {
    BitField bf = BitField::zeros();
    bf.set_bit(i);
    EXPECT_EQ(bf.bit_scan_after(0), i);
  }
}

TEST(LlvmLibcFlatTlsfTest, BitFieldBsfFromIndex) {
  for (uint32_t i = 0; i < BitField::BITS; ++i) {
    BitField bf = BitField::zeros();

    for (uint32_t j = i; j < BitField::BITS; ++j)
      bf.set_bit(j);

    for (uint32_t j = 0; j < BitField::BITS; ++j)
      EXPECT_EQ(bf.bit_scan_after(j), LIBC_NAMESPACE::cpp::max(i, j));
  }
}

TEST(LlvmLibcFlatTlsfTest, BitFieldBsfFromIndex1) {
  BitField bf = BitField::zeros();
  bf.set_bit(0);
  for (uint32_t i = 1; i < BitField::BITS; ++i)
    EXPECT_EQ(bf.bit_scan_after(i), static_cast<uint32_t>(BitField::BITS));
}

TEST(LlvmLibcFlatTlsfTest, BitFieldBsfFirstLast) {
  BitField bf = BitField::zeros();
  bf.set_bit(0);
  bf.set_bit(BitField::BITS - 1);

  EXPECT_EQ(bf.bit_scan_after(0), 0u);
  for (uint32_t i = 1; i < BitField::BITS; ++i)
    EXPECT_EQ(bf.bit_scan_after(i), static_cast<uint32_t>(BitField::BITS - 1));
}

TEST(LlvmLibcFlatTlsfTest, BitFieldBsfOneBehind) {
  for (uint32_t i = 1; i < BitField::BITS; ++i) {
    BitField bf = BitField::zeros();
    bf.set_bit(i - 1);
    EXPECT_EQ(bf.bit_scan_after(i), static_cast<uint32_t>(BitField::BITS));
  }
}

TEST(LlvmLibcFlatTlsfTest, BitFieldBsfOneBehindOneForward) {
  for (uint32_t i = 1; i < BitField::BITS - 1; ++i) {
    BitField bf = BitField::zeros();
    bf.set_bit(i - 1);
    bf.set_bit(i + 1);
    EXPECT_EQ(bf.bit_scan_after(i), i + 1);
  }
}

TEST(LlvmLibcFlatTlsfTest, BitFieldBsfOneBehindOneForwardOneOnPoint) {
  for (uint32_t i = 1; i < BitField::BITS - 1; ++i) {
    BitField bf = BitField::zeros();
    bf.set_bit(i - 1);
    bf.set_bit(i);
    bf.set_bit(i + 1);
    EXPECT_EQ(bf.bit_scan_after(i), i);
  }
}

TEST(LlvmLibcFlatTlsfTest, BitFieldBsfOneForward) {
  for (uint32_t i = 0; i < BitField::BITS - 1; ++i) {
    BitField bf = BitField::zeros();
    bf.set_bit(i + 1);
    EXPECT_EQ(bf.bit_scan_after(i), i + 1);
  }
}

TEST(LlvmLibcFlatTlsfTest, BitFieldBsfOnes) {
  for (uint32_t i = 0; i < BitField::BITS; ++i) {
    BitField bf = BitField::zeros();

    for (uint32_t j = i; j < BitField::BITS; ++j)
      bf.set_bit(j);
    EXPECT_EQ(bf.bit_scan_after(0), i);
  }
}

TEST(LlvmLibcFlatTlsfTest, BitFieldBsfOnesBelow) {
  for (uint32_t i = 0; i < BitField::BITS; ++i) {
    BitField bf = BitField::zeros();
    for (uint32_t j = 0; j < i; ++j)
      bf.set_bit(j);

    EXPECT_EQ(bf.bit_scan_after(i), static_cast<uint32_t>(BitField::BITS));
  }
}

} // namespace
