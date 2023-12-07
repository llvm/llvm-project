//===-- Unittests for byte ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/cstddef.h"
#include "test/UnitTest/Test.h"

namespace __llvm_libc::cpp {

TEST(LlvmLibcByteTest, to_integer) {
  const char str[] = "abc";
  const byte *const ptr = reinterpret_cast<const byte *>(str);
  ASSERT_EQ(to_integer<char>(ptr[0]), 'a');
  ASSERT_EQ(to_integer<char>(ptr[1]), 'b');
  ASSERT_EQ(to_integer<char>(ptr[2]), 'c');
  ASSERT_EQ(to_integer<char>(ptr[3]), '\0');
}

TEST(LlvmLibcByteTest, bitwise) {
  byte b{42};
  ASSERT_EQ(b, byte{0b00101010});

  b <<= 1;
  ASSERT_EQ(b, byte{0b01010100});
  b >>= 1;

  ASSERT_EQ((b << 1), byte{0b01010100});
  ASSERT_EQ((b >> 1), byte{0b00010101});

  b |= byte{0b11110000};
  ASSERT_EQ(b, byte{0b11111010});

  b &= byte{0b11110000};
  ASSERT_EQ(b, byte{0b11110000});

  b ^= byte{0b11111111};
  ASSERT_EQ(b, byte{0b00001111});
}

} // namespace __llvm_libc::cpp
