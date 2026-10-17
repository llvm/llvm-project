//===-- Unittests for character_converter utf8->utf16 ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/properties/types.h"
#include "src/__support/wchar/character_converter.h"
#include "src/__support/wchar/mbstate.h"
#include "test/UnitTest/Test.h"

#if defined(LIBC_TYPES_WCHAR_T_IS_UTF16)
using TestCharTypesUTF16 = LIBC_NAMESPACE::testing::TypeList<char16_t, wchar_t>;
#else
using TestCharTypesUTF16 = LIBC_NAMESPACE::testing::TypeList<char16_t>;
#endif

TYPED_TEST(LlvmLibcCharacterConverterUTF8To16Test, OneByte,
           TestCharTypesUTF16) {
  using CharType16 = ParamType;

  LIBC_NAMESPACE::internal::mbstate state;
  state.bytes_stored = 0;
  state.total_bytes = 0;
  char ch = 'A';

  LIBC_NAMESPACE::internal::CharacterConverter char_conv(&state);
  int err = char_conv.push(static_cast<char8_t>(ch));
  auto wch = char_conv.pop<CharType16>();

  ASSERT_EQ(err, 0);
  ASSERT_TRUE(wch.has_value());
  ASSERT_EQ(static_cast<int>(wch.value()), 65);
}

TYPED_TEST(LlvmLibcCharacterConverterUTF8To16Test, TwoBytes,
           TestCharTypesUTF16) {
  using CharType16 = ParamType;

  LIBC_NAMESPACE::internal::mbstate state;
  state.bytes_stored = 0;
  state.total_bytes = 0;
  const char ch[2] = {static_cast<char>(0xC2),
                      static_cast<char>(0x8E)}; //  car symbol

  LIBC_NAMESPACE::internal::CharacterConverter char_conv(&state);
  char_conv.push(static_cast<char8_t>(ch[0]));
  ASSERT_FALSE(char_conv.isPartiallyPopping());
  char_conv.push(static_cast<char8_t>(ch[1]));
  auto wch = char_conv.pop<CharType16>();

  ASSERT_TRUE(wch.has_value());
  ASSERT_EQ(static_cast<int>(wch.value()), 142);
}

TYPED_TEST(LlvmLibcCharacterConverterUTF8To16Test, ThreeBytes,
           TestCharTypesUTF16) {
  using CharType16 = ParamType;

  LIBC_NAMESPACE::internal::mbstate state;
  state.bytes_stored = 0;
  state.total_bytes = 0;
  const char ch[3] = {static_cast<char>(0xE2), static_cast<char>(0x88),
                      static_cast<char>(0x91)}; // ∑ sigma symbol

  LIBC_NAMESPACE::internal::CharacterConverter char_conv(&state);
  char_conv.push(static_cast<char8_t>(ch[0]));
  ASSERT_FALSE(char_conv.isPartiallyPopping());
  char_conv.push(static_cast<char8_t>(ch[1]));
  ASSERT_FALSE(char_conv.isPartiallyPopping());
  char_conv.push(static_cast<char8_t>(ch[2]));
  auto wch = char_conv.pop<CharType16>();

  ASSERT_TRUE(wch.has_value());
  ASSERT_EQ(static_cast<int>(wch.value()), 8721);
}

TYPED_TEST(LlvmLibcCharacterConverterUTF8To16Test, FourBytes,
           TestCharTypesUTF16) {
  using CharType16 = ParamType;

  LIBC_NAMESPACE::internal::mbstate state;
  state.bytes_stored = 0;
  state.total_bytes = 0;
  const char ch[4] = {static_cast<char>(0xF0), static_cast<char>(0x9F),
                      static_cast<char>(0xA4),
                      static_cast<char>(0xA1)}; // 🤡 clown emoji

  LIBC_NAMESPACE::internal::CharacterConverter char_conv(&state);
  char_conv.push(static_cast<char8_t>(ch[0]));
  ASSERT_FALSE(char_conv.isPartiallyPopping());
  char_conv.push(static_cast<char8_t>(ch[1]));
  ASSERT_FALSE(char_conv.isPartiallyPopping());
  char_conv.push(static_cast<char8_t>(ch[2]));
  ASSERT_FALSE(char_conv.isPartiallyPopping());
  char_conv.push(static_cast<char8_t>(ch[3]));
  auto wch_h = char_conv.pop<CharType16>();
  ASSERT_TRUE(char_conv.isPartiallyPopping());
  auto wch_l = char_conv.pop<CharType16>();

  ASSERT_TRUE(wch_h.has_value());
  ASSERT_EQ(static_cast<int>(wch_h.value()), 55358);
  ASSERT_TRUE(wch_l.has_value());
  ASSERT_EQ(static_cast<int>(wch_l.value()), 56609);
}

TYPED_TEST(LlvmLibcCharacterConverterUTF8To16Test, ValidTwoByteWithExtraRead,
           TestCharTypesUTF16) {
  using CharType16 = ParamType;

  LIBC_NAMESPACE::internal::mbstate state;
  state.bytes_stored = 0;
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
  ASSERT_EQ(err, EILSEQ);

  // Should produce an error since mbstate was reset
  auto wch = char_conv.pop<CharType16>();
  ASSERT_FALSE(wch.has_value());
}

TYPED_TEST(LlvmLibcCharacterConverterUTF8To16Test, TwoValidTwoBytes,
           TestCharTypesUTF16) {
  using CharType16 = ParamType;

  LIBC_NAMESPACE::internal::mbstate state;
  state.bytes_stored = 0;
  state.total_bytes = 0;
  const char ch[4] = {static_cast<char>(0xC2), static_cast<char>(0x8E),
                      static_cast<char>(0xC7), static_cast<char>(0x8C)};

  LIBC_NAMESPACE::internal::CharacterConverter char_conv(&state);
  int err = char_conv.push(static_cast<char8_t>(ch[0]));
  ASSERT_EQ(err, 0);
  err = char_conv.push(static_cast<char8_t>(ch[1]));
  ASSERT_EQ(err, 0);
  auto wch = char_conv.pop<CharType16>();
  ASSERT_TRUE(wch.has_value());
  ASSERT_EQ(static_cast<int>(wch.value()), 142);

  // Second two byte character
  err = char_conv.push(static_cast<char8_t>(ch[2]));
  ASSERT_EQ(err, 0);
  err = char_conv.push(static_cast<char8_t>(ch[3]));
  ASSERT_EQ(err, 0);
  wch = char_conv.pop<CharType16>();
  ASSERT_TRUE(wch.has_value());
  ASSERT_EQ(static_cast<int>(wch.value()), 460);
}

TYPED_TEST(LlvmLibcCharacterConverterUTF8To16Test, InvalidPop,
           TestCharTypesUTF16) {
  using CharType16 = ParamType;

  LIBC_NAMESPACE::internal::mbstate state;
  state.bytes_stored = 0;
  state.total_bytes = 0;
  LIBC_NAMESPACE::internal::CharacterConverter char_conv(&state);
  const char ch[2] = {static_cast<char>(0xC2), static_cast<char>(0x8E)};
  int err = char_conv.push(static_cast<char8_t>(ch[0]));
  ASSERT_EQ(err, 0);
  auto wch = char_conv.pop<CharType16>();
  ASSERT_FALSE(
      wch.has_value()); // Should fail since we have not read enough bytes
  err = char_conv.push(static_cast<char8_t>(ch[1]));
  ASSERT_EQ(err, 0);
  wch = char_conv.pop<CharType16>();
  ASSERT_TRUE(wch.has_value());
  ASSERT_EQ(static_cast<int>(wch.value()), 142);
}
