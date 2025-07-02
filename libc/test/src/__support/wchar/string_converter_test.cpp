//===-- Unittests for StringConverter class -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/types/char32_t.h"
#include "hdr/types/char8_t.h"
#include "src/__support/error_or.h"
#include "src/__support/wchar/mbstate.h"
#include "src/__support/wchar/string_converter.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStringConverterTest, UTF8To32) {
  // first 4 bytes are clown emoji (🤡), then next 3 are sigma symbol (∑)
  const char *src = "\xF0\x9F\xA4\xA1\xE2\x88\x91";
  LIBC_NAMESPACE::internal::mbstate state;
  LIBC_NAMESPACE::internal::StringConverter<char8_t> sc(
      reinterpret_cast<const char8_t *>(src), SIZE_MAX, &state);

  auto res = sc.popUTF32();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0x1f921);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 4);

  res = sc.popUTF32();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0x2211);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 7);

  res = sc.popUTF32();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 8);

  res = sc.popUTF32();
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(res.error(), -1);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 8);
}

TEST(LlvmLibcStringConverterTest, UTF32To8) {
  const wchar_t *src = L"\x1f921\x2211"; // clown emoji, sigma symbol
  LIBC_NAMESPACE::internal::mbstate state;
  LIBC_NAMESPACE::internal::StringConverter<char32_t> sc(
      reinterpret_cast<const char32_t *>(src), SIZE_MAX, &state);

  auto res = sc.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xF0);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 0);

  res = sc.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0x9F);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 0);

  res = sc.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xA4);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 0);

  res = sc.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xA1);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 1);

  res = sc.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xE2);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 1);

  res = sc.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0x88);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 1);

  res = sc.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0x91);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 2);

  res = sc.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 3);

  res = sc.popUTF8();
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(res.error(), -1);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 3);
}

TEST(LlvmLibcStringConverterTest, UTF32To8PartialRead) {
  const wchar_t *src = L"\x1f921\x2211"; // clown emoji, sigma symbol
  LIBC_NAMESPACE::internal::mbstate state;
  LIBC_NAMESPACE::internal::StringConverter<char32_t> sc(
      reinterpret_cast<const char32_t *>(src), 1, SIZE_MAX, &state);

  auto res = sc.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xF0);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 0);

  res = sc.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0x9F);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 0);

  res = sc.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xA4);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 0);

  res = sc.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xA1);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 1);

  res = sc.popUTF8();
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(res.error(), -1);
}

TEST(LlvmLibcStringConverterTest, UTF8To32PartialRead) {
  // first 4 bytes are clown emoji, then next 3 are sigma symbol
  const char *src = "\xF0\x9F\xA4\xA1\xE2\x88\x91";
  LIBC_NAMESPACE::internal::mbstate state;
  LIBC_NAMESPACE::internal::StringConverter<char8_t> sc(
      reinterpret_cast<const char8_t *>(src), 5, SIZE_MAX, &state);

  auto res = sc.popUTF32();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0x1f921);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 4);

  res = sc.popUTF32();
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.error()), -1);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 5);
}

TEST(LlvmLibcStringConverterTest, UTF32To8ErrorHandling) {
  const wchar_t *src = L"\x1f921\xffffff"; // clown emoji, invalid utf32
  LIBC_NAMESPACE::internal::mbstate state;
  LIBC_NAMESPACE::internal::StringConverter<char32_t> sc(
      reinterpret_cast<const char32_t *>(src), SIZE_MAX, &state);

  auto res = sc.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xF0);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 0);

  res = sc.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0x9F);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 0);

  res = sc.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xA4);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 0);

  res = sc.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xA1);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 1);

  res = sc.popUTF8();
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.error()), EILSEQ);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 1);
}

TEST(LlvmLibcStringConverterTest, UTF8To32ErrorHandling) {
  // first 4 bytes are clown emoji (🤡)
  // next 3 form an invalid character
  const char *src = "\xF0\x9F\xA4\xA1\x90\x88\x30";
  LIBC_NAMESPACE::internal::mbstate state;
  LIBC_NAMESPACE::internal::StringConverter<char8_t> sc(
      reinterpret_cast<const char8_t *>(src), SIZE_MAX, &state);

  auto res = sc.popUTF32();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0x1f921);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 4);

  res = sc.popUTF32();
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.error()), EILSEQ);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 4);
}

TEST(LlvmLibcStringConverterTest, MultipleStringConverters32To8) {
  /*
  We do NOT test partially popping a character and expecting the next
  StringConverter to continue where we left off. This is not expected to work
  and considered invalid.
  */
  const wchar_t *src = L"\x1f921\xff"; // clown emoji, sigma symbol
  LIBC_NAMESPACE::internal::mbstate state;
  LIBC_NAMESPACE::internal::StringConverter<char32_t> sc1(
      reinterpret_cast<const char32_t *>(src), 1, SIZE_MAX, &state);

  auto res = sc1.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xF0);
  ASSERT_EQ(static_cast<int>(sc1.getSourceIndex()), 0);

  res = sc1.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0x9F);
  ASSERT_EQ(static_cast<int>(sc1.getSourceIndex()), 0);

  res = sc1.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xA4);
  ASSERT_EQ(static_cast<int>(sc1.getSourceIndex()), 0);

  res = sc1.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xA1);
  ASSERT_EQ(static_cast<int>(sc1.getSourceIndex()), 1);

  LIBC_NAMESPACE::internal::StringConverter<char32_t> sc2(
      reinterpret_cast<const char32_t *>(src) + sc1.getSourceIndex(), 1,
      SIZE_MAX, &state);

  res = sc2.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xC3);
  ASSERT_EQ(static_cast<int>(sc2.getSourceIndex()), 0);

  res = sc2.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xBF);
  ASSERT_EQ(static_cast<int>(sc2.getSourceIndex()), 1);
}

TEST(LlvmLibcStringConverterTest, MultipleStringConverters8To32) {
  const char *src = "\xF0\x9F\xA4\xA1"; // clown emoji
  LIBC_NAMESPACE::internal::mbstate state;
  LIBC_NAMESPACE::internal::StringConverter<char8_t> sc1(
      reinterpret_cast<const char8_t *>(src), 2, SIZE_MAX, &state);

  auto res = sc1.popUTF32();
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.error()), -1);
  ASSERT_EQ(static_cast<int>(sc1.getSourceIndex()), 2);

  LIBC_NAMESPACE::internal::StringConverter<char8_t> sc2(
      reinterpret_cast<const char8_t *>(src) + sc1.getSourceIndex(), 3,
      SIZE_MAX, &state);

  res = sc2.popUTF32();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0x1f921);
  ASSERT_EQ(static_cast<int>(sc2.getSourceIndex()), 2);

  res = sc2.popUTF32();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0);
  ASSERT_EQ(static_cast<int>(sc2.getSourceIndex()), 3);
}

TEST(LlvmLibcStringConverterTest, DstLimitUTF8To32) {
  const char *src = "\xF0\x9F\xA4\xA1\xF0\x9F\xA4\xA1"; // 2 clown emojis
  LIBC_NAMESPACE::internal::mbstate state;
  LIBC_NAMESPACE::internal::StringConverter<char8_t> sc(
      reinterpret_cast<const char8_t *>(src), SIZE_MAX, 1, &state);

  auto res = sc.popUTF32();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 4);

  res = sc.popUTF32(); // no space to pop this into
  ASSERT_FALSE(res.has_value());
}

TEST(LlvmLibcStringConverterTest, DstLimitUTF32To8) {
  const wchar_t *src = L"\x1f921\x1f921"; // 2 clown emojis
  LIBC_NAMESPACE::internal::mbstate state;
  LIBC_NAMESPACE::internal::StringConverter<char32_t> sc(
      reinterpret_cast<const char32_t *>(src), SIZE_MAX, 5, &state);

  auto res = sc.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 0);

  res = sc.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 0);

  res = sc.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 0);

  res = sc.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 1);

  res = sc.popUTF8();
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 1);
}
