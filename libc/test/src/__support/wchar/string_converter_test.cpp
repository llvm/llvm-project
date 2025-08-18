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
#include "src/__support/macros/properties/os.h"
#include "src/__support/wchar/mbstate.h"
#include "src/__support/wchar/string_converter.h"
#include "test/UnitTest/Test.h"

// TODO: add support for 16-bit widechars to StringConverter to remove this
// macro
#ifdef LIBC_TARGET_OS_IS_WINDOWS
TEST(LlvmLibcStringConverterTest, Windows) {
  // pass on windows for now
}

#else

TEST(LlvmLibcStringConverterTest, UTF8To32) {
  // first 4 bytes are clown emoji (🤡)
  // next 3 bytes are sigma symbol (∑)
  // next 2 bytes are y with diaeresis (ÿ)
  // last byte is the letter A
  const char *src = "\xF0\x9F\xA4\xA1\xE2\x88\x91\xC3\xBF\x41";
  LIBC_NAMESPACE::internal::mbstate state;
  LIBC_NAMESPACE::internal::StringConverter<char8_t> sc(
      reinterpret_cast<const char8_t *>(src), &state, SIZE_MAX);

  auto res = sc.pop<char32_t>();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0x1f921);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 4);

  res = sc.pop<char32_t>();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0x2211);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 7);

  res = sc.pop<char32_t>();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xff);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 9);

  res = sc.pop<char32_t>();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0x41);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 10);

  res = sc.pop<char32_t>();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 11);

  res = sc.pop<char32_t>();
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(res.error(), -1);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 11);
}

TEST(LlvmLibcStringConverterTest, UTF32To8) {
  // clown emoji, sigma symbol, y with diaeresis, letter A
  const wchar_t src[] = {static_cast<wchar_t>(0x1f921),
                         static_cast<wchar_t>(0x2211),
                         static_cast<wchar_t>(0xff), static_cast<wchar_t>(0x41),
                         static_cast<wchar_t>(0x0)};
  LIBC_NAMESPACE::internal::mbstate state;
  LIBC_NAMESPACE::internal::StringConverter<char32_t> sc(
      reinterpret_cast<const char32_t *>(src), &state, SIZE_MAX);

  auto res = sc.pop<char8_t>();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xF0);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 1);

  res = sc.pop<char8_t>();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0x9F);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 1);

  res = sc.pop<char8_t>();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xA4);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 1);

  res = sc.pop<char8_t>();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xA1);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 1);

  // end of clown emoji, sigma symbol begins
  res = sc.pop<char8_t>();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xE2);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 2);

  res = sc.pop<char8_t>();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0x88);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 2);

  res = sc.pop<char8_t>();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0x91);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 2);

  // end of sigma symbol, y with diaeresis begins
  res = sc.pop<char8_t>();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xC3);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 3);

  res = sc.pop<char8_t>();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xBF);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 3);

  // end of y with diaeresis, letter A begins
  res = sc.pop<char8_t>();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0x41);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 4);

  // null byte
  res = sc.pop<char8_t>();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 5);

  res = sc.pop<char8_t>();
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(res.error(), -1);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 5);
}

TEST(LlvmLibcStringConverterTest, UTF32To8PartialRead) {
  const wchar_t src[] = {
      static_cast<wchar_t>(0x1f921), static_cast<wchar_t>(0x2211),
      static_cast<wchar_t>(0x0)}; // clown emoji, sigma symbol
  LIBC_NAMESPACE::internal::mbstate state;
  LIBC_NAMESPACE::internal::StringConverter<char32_t> sc(
      reinterpret_cast<const char32_t *>(src), &state, SIZE_MAX, 1);

  auto res = sc.pop<char8_t>();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xF0);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 1);

  res = sc.pop<char8_t>();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0x9F);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 1);

  res = sc.pop<char8_t>();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xA4);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 1);

  res = sc.pop<char8_t>();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xA1);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 1);

  // can only read 1 character from source string, so error on next pop
  res = sc.pop<char8_t>();
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(res.error(), -1);
}

TEST(LlvmLibcStringConverterTest, UTF8To32PartialRead) {
  // first 4 bytes are clown emoji, then next 3 are sigma symbol
  const char *src = "\xF0\x9F\xA4\xA1\xE2\x88\x91";
  LIBC_NAMESPACE::internal::mbstate state;
  LIBC_NAMESPACE::internal::StringConverter<char8_t> sc(
      reinterpret_cast<const char8_t *>(src), &state, SIZE_MAX, 5);

  auto res = sc.pop<char32_t>();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0x1f921);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 4);

  res = sc.pop<char32_t>();
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.error()), -1);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 5);
}

TEST(LlvmLibcStringConverterTest, UTF32To8ErrorHandling) {
  const wchar_t src[] = {
      static_cast<wchar_t>(0x1f921), static_cast<wchar_t>(0xffffff),
      static_cast<wchar_t>(0x0)}; // clown emoji, invalid utf32
  LIBC_NAMESPACE::internal::mbstate state;
  LIBC_NAMESPACE::internal::StringConverter<char32_t> sc(
      reinterpret_cast<const char32_t *>(src), &state, SIZE_MAX);

  auto res = sc.pop<char8_t>();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xF0);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 1);

  res = sc.pop<char8_t>();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0x9F);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 1);

  res = sc.pop<char8_t>();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xA4);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 1);

  res = sc.pop<char8_t>();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xA1);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 1);

  res = sc.pop<char8_t>();
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
      reinterpret_cast<const char8_t *>(src), &state, SIZE_MAX);

  auto res = sc.pop<char32_t>();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0x1f921);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 4);

  res = sc.pop<char32_t>();
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.error()), EILSEQ);
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 4);
}

TEST(LlvmLibcStringConverterTest, InvalidCharacterOutsideBounds) {
  // if an invalid character exists in the source string but we don't have space
  // to write it, we should return a "stop converting" error rather than an
  // invalid character error

  // first 4 bytes are clown emoji (🤡)
  // next 3 form an invalid character
  const char *src1 = "\xF0\x9F\xA4\xA1\x90\x88\x30";
  LIBC_NAMESPACE::internal::mbstate ps1;
  LIBC_NAMESPACE::internal::StringConverter<char8_t> sc1(
      reinterpret_cast<const char8_t *>(src1), &ps1, 1);

  auto res1 = sc1.pop<char32_t>();
  ASSERT_TRUE(res1.has_value());
  ASSERT_EQ(static_cast<int>(res1.value()), 0x1f921);
  ASSERT_EQ(static_cast<int>(sc1.getSourceIndex()), 4);

  res1 = sc1.pop<char32_t>();
  ASSERT_FALSE(res1.has_value());
  // no space to write error NOT invalid character error (EILSEQ)
  ASSERT_EQ(static_cast<int>(res1.error()), -1);
  ASSERT_EQ(static_cast<int>(sc1.getSourceIndex()), 4);

  const wchar_t src2[] = {
      static_cast<wchar_t>(0x1f921), static_cast<wchar_t>(0xffffff),
      static_cast<wchar_t>(0x0)}; // clown emoji, invalid utf32
  LIBC_NAMESPACE::internal::mbstate ps2;
  LIBC_NAMESPACE::internal::StringConverter<char32_t> sc2(
      reinterpret_cast<const char32_t *>(src2), &ps2, 4);

  auto res2 = sc2.pop<char8_t>();
  ASSERT_TRUE(res2.has_value());
  ASSERT_EQ(static_cast<int>(res2.value()), 0xF0);
  ASSERT_EQ(static_cast<int>(sc2.getSourceIndex()), 1);

  res2 = sc2.pop<char8_t>();
  ASSERT_TRUE(res2.has_value());
  ASSERT_EQ(static_cast<int>(res2.value()), 0x9F);
  ASSERT_EQ(static_cast<int>(sc2.getSourceIndex()), 1);

  res2 = sc2.pop<char8_t>();
  ASSERT_TRUE(res2.has_value());
  ASSERT_EQ(static_cast<int>(res2.value()), 0xA4);
  ASSERT_EQ(static_cast<int>(sc2.getSourceIndex()), 1);

  res2 = sc2.pop<char8_t>();
  ASSERT_TRUE(res2.has_value());
  ASSERT_EQ(static_cast<int>(res2.value()), 0xA1);
  ASSERT_EQ(static_cast<int>(sc2.getSourceIndex()), 1);

  res2 = sc2.pop<char8_t>();
  ASSERT_FALSE(res2.has_value());
  // no space to write error NOT invalid character error (EILSEQ)
  ASSERT_EQ(static_cast<int>(res2.error()), -1);
  ASSERT_EQ(static_cast<int>(sc2.getSourceIndex()), 1);
}

TEST(LlvmLibcStringConverterTest, MultipleStringConverters32To8) {
  /*
  We do NOT test partially popping a character and expecting the next
  StringConverter to continue where we left off. This is not expected to work
  and considered invalid.
  */
  const wchar_t src[] = {
      static_cast<wchar_t>(0x1f921), static_cast<wchar_t>(0xff),
      static_cast<wchar_t>(0x0)}; // clown emoji, y with diaeresis (ÿ)
  LIBC_NAMESPACE::internal::mbstate state;
  LIBC_NAMESPACE::internal::StringConverter<char32_t> sc1(
      reinterpret_cast<const char32_t *>(src), &state, SIZE_MAX, 1);

  auto res = sc1.pop<char8_t>();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xF0);
  ASSERT_EQ(static_cast<int>(sc1.getSourceIndex()), 1);

  res = sc1.pop<char8_t>();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0x9F);
  ASSERT_EQ(static_cast<int>(sc1.getSourceIndex()), 1);

  res = sc1.pop<char8_t>();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xA4);
  ASSERT_EQ(static_cast<int>(sc1.getSourceIndex()), 1);

  res = sc1.pop<char8_t>();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xA1);
  ASSERT_EQ(static_cast<int>(sc1.getSourceIndex()), 1);

  // sc2 should pick up where sc1 left off and continue the conversion
  LIBC_NAMESPACE::internal::StringConverter<char32_t> sc2(
      reinterpret_cast<const char32_t *>(src) + sc1.getSourceIndex(), &state,
      SIZE_MAX, 1);

  res = sc2.pop<char8_t>();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xC3);
  ASSERT_EQ(static_cast<int>(sc2.getSourceIndex()), 1);

  res = sc2.pop<char8_t>();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xBF);
  ASSERT_EQ(static_cast<int>(sc2.getSourceIndex()), 1);
}

TEST(LlvmLibcStringConverterTest, MultipleStringConverters8To32) {
  const char *src = "\xF0\x9F\xA4\xA1"; // clown emoji
  LIBC_NAMESPACE::internal::mbstate state;
  LIBC_NAMESPACE::internal::StringConverter<char8_t> sc1(
      reinterpret_cast<const char8_t *>(src), &state, SIZE_MAX, 2);

  auto res = sc1.pop<char32_t>();
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.error()), -1);
  ASSERT_EQ(static_cast<int>(sc1.getSourceIndex()), 2);

  // sc2 should pick up where sc1 left off and continue the conversion
  LIBC_NAMESPACE::internal::StringConverter<char8_t> sc2(
      reinterpret_cast<const char8_t *>(src) + sc1.getSourceIndex(), &state,
      SIZE_MAX, 3);

  res = sc2.pop<char32_t>();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0x1f921);
  ASSERT_EQ(static_cast<int>(sc2.getSourceIndex()), 2);

  res = sc2.pop<char32_t>();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0);
  ASSERT_EQ(static_cast<int>(sc2.getSourceIndex()), 3);
}

TEST(LlvmLibcStringConverterTest, DestLimitUTF8To32) {
  const char *src = "\xF0\x9F\xA4\xA1\xF0\x9F\xA4\xA1"; // 2 clown emojis
  LIBC_NAMESPACE::internal::mbstate state;
  LIBC_NAMESPACE::internal::StringConverter<char8_t> sc(
      reinterpret_cast<const char8_t *>(src), &state, 1);

  auto res = sc.pop<char32_t>();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 4);

  res = sc.pop<char32_t>(); // no space to pop this into
  ASSERT_FALSE(res.has_value());
}

TEST(LlvmLibcStringConverterTest, DestLimitUTF32To8) {
  const wchar_t src[] = {static_cast<wchar_t>(0x1f921),
                         static_cast<wchar_t>(0x1f921)}; // 2 clown emojis
  LIBC_NAMESPACE::internal::mbstate state;
  LIBC_NAMESPACE::internal::StringConverter<char32_t> sc(
      reinterpret_cast<const char32_t *>(src), &state, 5);

  auto res = sc.pop<char8_t>();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 1);

  res = sc.pop<char8_t>();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 1);

  res = sc.pop<char8_t>();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 1);

  res = sc.pop<char8_t>();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 1);

  res = sc.pop<char8_t>();
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(static_cast<int>(sc.getSourceIndex()), 1);
}

#endif
