//===-- Unittests for the CharacterConverter class (utf16 -> 8) -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/common.h"
#include "src/__support/macros/properties/types.h"
#include "src/__support/wchar/character_converter.h"
#include "src/__support/wchar/mbstate.h"

#include "test/UnitTest/Test.h"

#if defined(LIBC_TYPES_WCHAR_T_IS_UTF16)
using TestCharTypesUTF16 = LIBC_NAMESPACE::testing::TypeList<char16_t, wchar_t>;
#else
using TestCharTypesUTF16 = LIBC_NAMESPACE::testing::TypeList<char16_t>;
#endif

TYPED_TEST(LlvmLibcCharacterConverterUTF16To8Test, OneByte,
           TestCharTypesUTF16) {
  using CharType16 = ParamType;

  LIBC_NAMESPACE::internal::mbstate state;
  LIBC_NAMESPACE::internal::CharacterConverter cr(&state);
  cr.clear();

  // utf8 1-byte encodings are identical to their utf16 representations
  CharType16 utf16_A = 0x41; // 'A'
  cr.push(utf16_A);
  ASSERT_TRUE(cr.isFull());
  auto popped = cr.pop_utf8();
  ASSERT_TRUE(popped.has_value());
  ASSERT_EQ(static_cast<char>(popped.value()), 'A');
  ASSERT_TRUE(cr.isEmpty());

  CharType16 utf16_B = 0x42; // 'B'
  cr.push(utf16_B);
  ASSERT_TRUE(cr.isFull());
  popped = cr.pop_utf8();
  ASSERT_TRUE(popped.has_value());
  ASSERT_EQ(static_cast<char>(popped.value()), 'B');
  ASSERT_TRUE(cr.isEmpty());

  // should error if we try to pop another utf8 byte out
  popped = cr.pop_utf8();
  ASSERT_FALSE(popped.has_value());
}

TYPED_TEST(LlvmLibcCharacterConverterUTF16To8Test, TwoByte,
           TestCharTypesUTF16) {
  using CharType16 = ParamType;

  LIBC_NAMESPACE::internal::mbstate state;
  LIBC_NAMESPACE::internal::CharacterConverter cr(&state);
  cr.clear();

  // testing utf16: 0xff -> utf8: 0xc3 0xbf
  CharType16 utf16 = 0xff;
  cr.push(utf16);
  ASSERT_TRUE(cr.isFull());
  auto popped = cr.pop_utf8();
  ASSERT_TRUE(popped.has_value());
  ASSERT_EQ(static_cast<int>(popped.value()), 0xc3);
  ASSERT_TRUE(cr.isPartiallyPopping());
  popped = cr.pop_utf8();
  ASSERT_TRUE(popped.has_value());
  ASSERT_EQ(static_cast<int>(popped.value()), 0xbf);
  ASSERT_TRUE(cr.isEmpty());

  // testing utf16: 0x58e -> utf8: 0xd6 0x8e
  utf16 = 0x58e;
  cr.push(utf16);
  ASSERT_TRUE(cr.isFull());
  popped = cr.pop_utf8();
  ASSERT_TRUE(popped.has_value());
  ASSERT_EQ(static_cast<int>(popped.value()), 0xd6);
  ASSERT_TRUE(cr.isPartiallyPopping());
  popped = cr.pop_utf8();
  ASSERT_TRUE(popped.has_value());
  ASSERT_EQ(static_cast<int>(popped.value()), 0x8e);
  ASSERT_TRUE(cr.isEmpty());

  // should error if we try to pop another utf8 byte out
  popped = cr.pop_utf8();
  ASSERT_FALSE(popped.has_value());
}

TYPED_TEST(LlvmLibcCharacterConverterUTF16To8Test, ThreeByte,
           TestCharTypesUTF16) {
  using CharType16 = ParamType;

  LIBC_NAMESPACE::internal::mbstate state;
  LIBC_NAMESPACE::internal::CharacterConverter cr(&state);
  cr.clear();

  // testing utf16: 0xac15 -> utf8: 0xea 0xb0 0x95
  CharType16 utf16 = 0xac15;
  cr.push(utf16);
  ASSERT_TRUE(cr.isFull());
  auto popped = cr.pop_utf8();
  ASSERT_TRUE(popped.has_value());
  ASSERT_EQ(static_cast<int>(popped.value()), 0xea);
  ASSERT_TRUE(cr.isPartiallyPopping());
  popped = cr.pop_utf8();
  ASSERT_TRUE(popped.has_value());
  ASSERT_EQ(static_cast<int>(popped.value()), 0xb0);
  ASSERT_TRUE(cr.isPartiallyPopping());
  popped = cr.pop_utf8();
  ASSERT_TRUE(popped.has_value());
  ASSERT_EQ(static_cast<int>(popped.value()), 0x95);
  ASSERT_TRUE(cr.isEmpty());

  // testing utf16: 0x267b -> utf8: 0xe2 0x99 0xbb
  utf16 = 0x267b;
  cr.push(utf16);
  ASSERT_TRUE(cr.isFull());
  popped = cr.pop_utf8();
  ASSERT_TRUE(popped.has_value());
  ASSERT_EQ(static_cast<int>(popped.value()), 0xe2);
  ASSERT_TRUE(cr.isPartiallyPopping());
  popped = cr.pop_utf8();
  ASSERT_TRUE(popped.has_value());
  ASSERT_EQ(static_cast<int>(popped.value()), 0x99);
  ASSERT_TRUE(cr.isPartiallyPopping());
  popped = cr.pop_utf8();
  ASSERT_TRUE(popped.has_value());
  ASSERT_EQ(static_cast<int>(popped.value()), 0xbb);
  ASSERT_TRUE(cr.isEmpty());

  // should error if we try to pop another utf8 byte out
  popped = cr.pop_utf8();
  ASSERT_FALSE(popped.has_value());
}

TYPED_TEST(LlvmLibcCharacterConverterUTF16To8Test, FourByte,
           TestCharTypesUTF16) {
  using CharType16 = ParamType;

  LIBC_NAMESPACE::internal::mbstate state;
  LIBC_NAMESPACE::internal::CharacterConverter cr(&state);
  cr.clear();

  // testing utf16: 0xd83e 0xdd21 -> utf8: 0xf0 0x9f 0xa4 0xa1
  CharType16 utf16_h = 0xd83e;
  cr.push(utf16_h);
  ASSERT_FALSE(cr.isPartiallyPopping());
  CharType16 utf16_l = 0xdd21;
  cr.push(utf16_l);
  ASSERT_TRUE(cr.isFull());
  auto popped = cr.pop_utf8();
  ASSERT_TRUE(popped.has_value());
  ASSERT_EQ(static_cast<int>(popped.value()), 0xf0);
  ASSERT_TRUE(cr.isPartiallyPopping());
  popped = cr.pop_utf8();
  ASSERT_TRUE(popped.has_value());
  ASSERT_EQ(static_cast<int>(popped.value()), 0x9f);
  ASSERT_TRUE(cr.isPartiallyPopping());
  popped = cr.pop_utf8();
  ASSERT_TRUE(popped.has_value());
  ASSERT_EQ(static_cast<int>(popped.value()), 0xa4);
  ASSERT_TRUE(cr.isPartiallyPopping());
  popped = cr.pop_utf8();
  ASSERT_TRUE(popped.has_value());
  ASSERT_EQ(static_cast<int>(popped.value()), 0xa1);
  ASSERT_TRUE(cr.isEmpty());

  // testing utf16: 0xd808 0xdd21 -> utf8: 0xf0 0x92 0x84 0xa1
  utf16_h = 0xd808;
  cr.push(utf16_h);
  ASSERT_FALSE(cr.isPartiallyPopping());
  utf16_l = 0xdd21;
  cr.push(utf16_l);
  ASSERT_TRUE(cr.isFull());
  popped = cr.pop_utf8();
  ASSERT_TRUE(popped.has_value());
  ASSERT_EQ(static_cast<int>(popped.value()), 0xf0);
  ASSERT_TRUE(cr.isPartiallyPopping());
  popped = cr.pop_utf8();
  ASSERT_TRUE(popped.has_value());
  ASSERT_EQ(static_cast<int>(popped.value()), 0x92);
  ASSERT_TRUE(cr.isPartiallyPopping());
  popped = cr.pop_utf8();
  ASSERT_TRUE(popped.has_value());
  ASSERT_EQ(static_cast<int>(popped.value()), 0x84);
  ASSERT_TRUE(cr.isPartiallyPopping());
  popped = cr.pop_utf8();
  ASSERT_TRUE(popped.has_value());
  ASSERT_EQ(static_cast<int>(popped.value()), 0xa1);
  ASSERT_TRUE(cr.isEmpty());

  // should error if we try to pop another utf8 byte out
  popped = cr.pop_utf8();
  ASSERT_FALSE(popped.has_value());
}

TYPED_TEST(LlvmLibcCharacterConverterUTF16To8Test, CantPushMidConversion,
           TestCharTypesUTF16) {
  using CharType16 = ParamType;

  LIBC_NAMESPACE::internal::mbstate state;
  LIBC_NAMESPACE::internal::CharacterConverter cr(&state);
  cr.clear();

  // testing utf16: 0xd808 0xdd21 -> utf8: 0xf0 0x92 0x84 0xa1
  CharType16 utf16_h = 0xd808;
  ASSERT_EQ(cr.push(utf16_h), 0);
  CharType16 utf16_l = 0xdd21;
  ASSERT_EQ(cr.push(utf16_l), 0);
  auto popped = cr.pop_utf8();
  ASSERT_TRUE(popped.has_value());

  // can't push a utf16 without finishing popping the utf8 bytes out
  int err = cr.push(utf16_l);
  ASSERT_EQ(err, -1);
}

TYPED_TEST(LlvmLibcCharacterConverterUTF16To8Test, CantPopMidPushing,
           TestCharTypesUTF16) {
  using CharType16 = ParamType;

  LIBC_NAMESPACE::internal::mbstate state;
  LIBC_NAMESPACE::internal::CharacterConverter cr(&state);
  cr.clear();

  // testing utf16: 0xd808 0xdd21 -> utf8: 0xf0 0x92 0x84 0xa1
  CharType16 utf16_h = 0xd808;
  ASSERT_EQ(cr.push(utf16_h), 0);

  // can't pop a utf8 without finishing pushing the utf16 code units
  auto popped = cr.pop_utf8();
  ASSERT_FALSE(popped.has_value());
}
