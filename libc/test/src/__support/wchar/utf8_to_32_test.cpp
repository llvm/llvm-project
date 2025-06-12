//===-- Unittests for character_converter utf8->utf32 ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/wchar/character_converter.h"
#include "src/__support/wchar/mbstate.h"
#include "src/__support/wchar/utf_ret.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcCharacterConverterUTF8To32Test, OneByte) {
  LIBC_NAMESPACE::internal::mbstate state;
  state.bytes_processed = 0;
  state.total_bytes = 0;
  char ch = 'A';

  LIBC_NAMESPACE::internal::CharacterConverter char_conv(&state);
  int err = char_conv.push(static_cast<char8_t>(ch));
  LIBC_NAMESPACE::internal::utf_ret<char32_t> wch = char_conv.pop_utf32();

  EXPECT_EQ(err, 0);
  EXPECT_EQ(wch.error, 0);
  EXPECT_EQ(static_cast<int>(wch.out), 65);
}

TEST(LlvmLibcCharacterConverterUTF8To32Test, TwoBytes) {
  LIBC_NAMESPACE::internal::mbstate state;
  state.bytes_processed = 0;
  state.total_bytes = 0;
  const char ch[2] = {static_cast<char>(0xC2), static_cast<char>(0x8E)}; // 

  LIBC_NAMESPACE::internal::CharacterConverter char_conv(&state);
  char_conv.push(static_cast<char8_t>(ch[0]));
  char_conv.push(static_cast<char8_t>(ch[1]));
  LIBC_NAMESPACE::internal::utf_ret<char32_t> wch = char_conv.pop_utf32();

  ASSERT_EQ(wch.error, 0);
  ASSERT_EQ(static_cast<int>(wch.out), 142);
}

TEST(LlvmLibcCharacterConverterUTF8To32Test, ThreeBytes) {
  LIBC_NAMESPACE::internal::mbstate state;
  state.bytes_processed = 0;
  state.total_bytes = 0;
  const char ch[3] = {static_cast<char>(0xE2), static_cast<char>(0x88),
                      static_cast<char>(0x91)}; // ∑

  LIBC_NAMESPACE::internal::CharacterConverter char_conv(&state);
  char_conv.push(static_cast<char8_t>(ch[0]));
  char_conv.push(static_cast<char8_t>(ch[1]));
  char_conv.push(static_cast<char8_t>(ch[2]));
  LIBC_NAMESPACE::internal::utf_ret<char32_t> wch = char_conv.pop_utf32();

  ASSERT_EQ(wch.error, 0);
  ASSERT_EQ(static_cast<int>(wch.out), 8721);
}

TEST(LlvmLibcCharacterConverterUTF8To32Test, FourBytes) {
  LIBC_NAMESPACE::internal::mbstate state;
  state.bytes_processed = 0;
  state.total_bytes = 0;
  const char ch[4] = {static_cast<char>(0xF0), static_cast<char>(0x9F),
                      static_cast<char>(0xA4), static_cast<char>(0xA1)}; // 🤡

  LIBC_NAMESPACE::internal::CharacterConverter char_conv(&state);
  char_conv.push(static_cast<char8_t>(ch[0]));
  char_conv.push(static_cast<char8_t>(ch[1]));
  char_conv.push(static_cast<char8_t>(ch[2]));
  char_conv.push(static_cast<char8_t>(ch[3]));
  LIBC_NAMESPACE::internal::utf_ret<char32_t> wch = char_conv.pop_utf32();

  ASSERT_EQ(wch.error, 0);
  ASSERT_EQ(static_cast<int>(wch.out), 129313);
}

TEST(LlvmLibcCharacterConverterUTF8To32Test, InvalidByte) {
  LIBC_NAMESPACE::internal::mbstate state;
  state.bytes_processed = 0;
  state.total_bytes = 0;
  const char ch = static_cast<char>(0x80); // invalid starting bit sequence

  LIBC_NAMESPACE::internal::CharacterConverter char_conv(&state);
  int err = char_conv.push(static_cast<char8_t>(ch));

  ASSERT_EQ(err, -1);
}

TEST(LlvmLibcCharacterConverterUTF8To32Test, InvalidMultiByte) {
  LIBC_NAMESPACE::internal::mbstate state;
  state.bytes_processed = 0;
  state.total_bytes = 0;
  const char ch[4] = {
      static_cast<char>(0x80), static_cast<char>(0x00), static_cast<char>(0x80),
      static_cast<char>(0x00)}; // first, third, and last bytes are invalid

  LIBC_NAMESPACE::internal::CharacterConverter char_conv(&state);
  int err = char_conv.push(static_cast<char8_t>(ch[0]));
  ASSERT_EQ(err, -1);
  err = char_conv.push(static_cast<char8_t>(ch[1]));
  ASSERT_EQ(err, 0);
  // Prev byte was single byte so trying to read another should error.
  err = char_conv.push(static_cast<char8_t>(ch[2]));
  ASSERT_EQ(err, -1);
  err = char_conv.push(static_cast<char8_t>(ch[3]));
  ASSERT_EQ(err, -1);
}

TEST(LlvmLibcCharacterConverterUTF8To32Test, InvalidLastByte) {
  LIBC_NAMESPACE::internal::mbstate state;
  state.bytes_processed = 0;
  state.total_bytes = 0;
  const char ch[4] = {static_cast<char>(0xF1), static_cast<char>(0x80),
                      static_cast<char>(0x80),
                      static_cast<char>(0xC0)}; // invalid last byte

  LIBC_NAMESPACE::internal::CharacterConverter char_conv(&state);
  int err = char_conv.push(static_cast<char8_t>(ch[0]));
  ASSERT_EQ(err, 0);
  err = char_conv.push(static_cast<char8_t>(ch[1]));
  ASSERT_EQ(err, 0);
  err = char_conv.push(static_cast<char8_t>(ch[2]));
  ASSERT_EQ(err, 0);
  err = char_conv.push(static_cast<char8_t>(ch[3]));
  ASSERT_EQ(err, -1);
}

TEST(LlvmLibcCharacterConverterUTF8To32Test, ValidTwoByteWithExtraRead) {
  LIBC_NAMESPACE::internal::mbstate state;
  state.bytes_processed = 0;
  state.total_bytes = 0;
  const char ch[3] = {static_cast<char>(0xC2), static_cast<char>(0x8E),
                      static_cast<char>(0x80)};

  LIBC_NAMESPACE::internal::CharacterConverter char_conv(&state);
  int err = char_conv.push(static_cast<char8_t>(ch[0]));
  ASSERT_EQ(err, 0);
  err = char_conv.push(static_cast<char8_t>(ch[1]));
  ASSERT_EQ(err, 0);
  // Should produce an error on 3rd byte
  err = char_conv.push(static_cast<char8_t>(ch[2]));
  ASSERT_EQ(err, -1);

  LIBC_NAMESPACE::internal::utf_ret<char32_t> wch = char_conv.pop_utf32();
  ASSERT_EQ(wch.error, 0);
  // Should still output the correct result.
  ASSERT_EQ(static_cast<int>(wch.out), 142);
}

TEST(LlvmLibcCharacterConverterUTF8To32Test, TwoValidTwoBytes) {
  LIBC_NAMESPACE::internal::mbstate state;
  state.bytes_processed = 0;
  state.total_bytes = 0;
  const char ch[4] = {static_cast<char>(0xC2), static_cast<char>(0x8E),
                      static_cast<char>(0xC7), static_cast<char>(0x8C)};

  LIBC_NAMESPACE::internal::CharacterConverter char_conv(&state);
  int err = char_conv.push(static_cast<char8_t>(ch[0]));
  ASSERT_EQ(err, 0);
  err = char_conv.push(static_cast<char8_t>(ch[1]));
  ASSERT_EQ(err, 0);
  LIBC_NAMESPACE::internal::utf_ret<char32_t> wch = char_conv.pop_utf32();
  ASSERT_EQ(wch.error, 0);
  ASSERT_EQ(static_cast<int>(wch.out), 142);

  // Second two byte character
  err = char_conv.push(static_cast<char8_t>(ch[2]));
  ASSERT_EQ(err, 0);
  err = char_conv.push(static_cast<char8_t>(ch[3]));
  ASSERT_EQ(err, 0);
  wch = char_conv.pop_utf32();
  ASSERT_EQ(wch.error, 0);
  ASSERT_EQ(static_cast<int>(wch.out), 460);
}
