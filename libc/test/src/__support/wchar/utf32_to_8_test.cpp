//===-- Unittests for the CharacterConverter class (utf32 -> 8) -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/common.h"
#include "src/__support/wchar/character_converter.h"
#include "src/__support/wchar/mbstate.h"

#include "test/UnitTest/Test.h"

TEST(LlvmLibcCharacterConverterUTF32To8Test, OneByte) {
  LIBC_NAMESPACE::internal::mbstate state;
  LIBC_NAMESPACE::internal::CharacterConverter cr(&state);

  // utf8 1-byte encodings are identical to their utf32 representations
  char32_t utf32_A = 0x41; // 'A'
  cr.push(utf32_A);
  auto popped = cr.pop_utf8();
  ASSERT_EQ(popped.error, 0);
  ASSERT_EQ(static_cast<char>(popped.out), 'A');
  ASSERT_TRUE(cr.isComplete());

  char32_t utf32_B = 0x42; // 'B'
  cr.push(utf32_B);
  popped = cr.pop_utf8();
  ASSERT_EQ(popped.error, 0);
  ASSERT_EQ(static_cast<char>(popped.out), 'B');
  ASSERT_TRUE(cr.isComplete());

  // should error if we try to pop another utf8 byte out
  popped = cr.pop_utf8();
  ASSERT_NE(popped.error, 0);
}

TEST(LlvmLibcCharacterConverterUTF32To8Test, TwoByte) {
  LIBC_NAMESPACE::internal::mbstate state;
  LIBC_NAMESPACE::internal::CharacterConverter cr(&state);

  // testing utf32: 0xff -> utf8: 0xc3 0xbf
  char32_t utf32 = 0xff;
  cr.push(utf32);
  auto popped = cr.pop_utf8();
  ASSERT_EQ(popped.error, 0);
  ASSERT_EQ(static_cast<int>(popped.out), 0xc3);
  ASSERT_TRUE(!cr.isComplete());
  popped = cr.pop_utf8();
  ASSERT_EQ(popped.error, 0);
  ASSERT_EQ(static_cast<int>(popped.out), 0xbf);
  ASSERT_TRUE(cr.isComplete());

  // testing utf32: 0x58e -> utf8: 0xd6 0x8e
  utf32 = 0x58e;
  cr.push(utf32);
  popped = cr.pop_utf8();
  ASSERT_EQ(popped.error, 0);
  ASSERT_EQ(static_cast<int>(popped.out), 0xd6);
  ASSERT_TRUE(!cr.isComplete());
  popped = cr.pop_utf8();
  ASSERT_EQ(popped.error, 0);
  ASSERT_EQ(static_cast<int>(popped.out), 0x8e);
  ASSERT_TRUE(cr.isComplete());

  // should error if we try to pop another utf8 byte out
  popped = cr.pop_utf8();
  ASSERT_NE(popped.error, 0);
}

TEST(LlvmLibcCharacterConverterUTF32To8Test, ThreeByte) {
  LIBC_NAMESPACE::internal::mbstate state;
  LIBC_NAMESPACE::internal::CharacterConverter cr(&state);

  // testing utf32: 0xac15 -> utf8: 0xea 0xb0 0x95
  char32_t utf32 = 0xac15;
  cr.push(utf32);
  auto popped = cr.pop_utf8();
  ASSERT_EQ(popped.error, 0);
  ASSERT_EQ(static_cast<int>(popped.out), 0xea);
  ASSERT_TRUE(!cr.isComplete());
  popped = cr.pop_utf8();
  ASSERT_EQ(popped.error, 0);
  ASSERT_EQ(static_cast<int>(popped.out), 0xb0);
  ASSERT_TRUE(!cr.isComplete());
  popped = cr.pop_utf8();
  ASSERT_EQ(popped.error, 0);
  ASSERT_EQ(static_cast<int>(popped.out), 0x95);
  ASSERT_TRUE(cr.isComplete());

  // testing utf32: 0x267b -> utf8: 0xe2 0x99 0xbb
  utf32 = 0x267b;
  cr.push(utf32);
  popped = cr.pop_utf8();
  ASSERT_EQ(popped.error, 0);
  ASSERT_EQ(static_cast<int>(popped.out), 0xe2);
  ASSERT_TRUE(!cr.isComplete());
  popped = cr.pop_utf8();
  ASSERT_EQ(popped.error, 0);
  ASSERT_EQ(static_cast<int>(popped.out), 0x99);
  ASSERT_TRUE(!cr.isComplete());
  popped = cr.pop_utf8();
  ASSERT_EQ(popped.error, 0);
  ASSERT_EQ(static_cast<int>(popped.out), 0xbb);
  ASSERT_TRUE(cr.isComplete());

  // should error if we try to pop another utf8 byte out
  popped = cr.pop_utf8();
  ASSERT_NE(popped.error, 0);
}

TEST(LlvmLibcCharacterConverterUTF32To8Test, FourByte) {
  LIBC_NAMESPACE::internal::mbstate state;
  LIBC_NAMESPACE::internal::CharacterConverter cr(&state);

  // testing utf32: 0x1f921 -> utf8: 0xf0 0x9f 0xa4 0xa1
  char32_t utf32 = 0x1f921;
  cr.push(utf32);
  auto popped = cr.pop_utf8();
  ASSERT_EQ(popped.error, 0);
  ASSERT_EQ(static_cast<int>(popped.out), 0xf0);
  ASSERT_TRUE(!cr.isComplete());
  popped = cr.pop_utf8();
  ASSERT_EQ(popped.error, 0);
  ASSERT_EQ(static_cast<int>(popped.out), 0x9f);
  ASSERT_TRUE(!cr.isComplete());
  popped = cr.pop_utf8();
  ASSERT_EQ(popped.error, 0);
  ASSERT_EQ(static_cast<int>(popped.out), 0xa4);
  ASSERT_TRUE(!cr.isComplete());
  popped = cr.pop_utf8();
  ASSERT_EQ(popped.error, 0);
  ASSERT_EQ(static_cast<int>(popped.out), 0xa1);
  ASSERT_TRUE(cr.isComplete());

  // testing utf32: 0x12121 -> utf8: 0xf0 0x92 0x84 0xa1
  utf32 = 0x12121;
  cr.push(utf32);
  popped = cr.pop_utf8();
  ASSERT_EQ(popped.error, 0);
  ASSERT_EQ(static_cast<int>(popped.out), 0xf0);
  ASSERT_TRUE(!cr.isComplete());
  popped = cr.pop_utf8();
  ASSERT_EQ(popped.error, 0);
  ASSERT_EQ(static_cast<int>(popped.out), 0x92);
  ASSERT_TRUE(!cr.isComplete());
  popped = cr.pop_utf8();
  ASSERT_EQ(popped.error, 0);
  ASSERT_EQ(static_cast<int>(popped.out), 0x84);
  ASSERT_TRUE(!cr.isComplete());
  popped = cr.pop_utf8();
  ASSERT_EQ(popped.error, 0);
  ASSERT_EQ(static_cast<int>(popped.out), 0xa1);
  ASSERT_TRUE(cr.isComplete());

  // should error if we try to pop another utf8 byte out
  popped = cr.pop_utf8();
  ASSERT_NE(popped.error, 0);
}
