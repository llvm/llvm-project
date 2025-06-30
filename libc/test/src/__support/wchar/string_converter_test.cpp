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
      reinterpret_cast<const char8_t *>(src), &state);

  auto res = sc.popUTF32();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0x1f921);

  res = sc.popUTF32();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0x2211);

  res = sc.popUTF32();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0);

  res = sc.popUTF32();
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(res.error(), -1);
}

TEST(LlvmLibcStringConverterTest, UTF32To8) {
  const wchar_t *src = L"\x1f921\x2211"; // clown emoji, sigma symbol
  LIBC_NAMESPACE::internal::mbstate state;
  LIBC_NAMESPACE::internal::StringConverter<char32_t> sc(
      reinterpret_cast<const char32_t *>(src), &state);

  auto res = sc.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xF0);

  res = sc.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0x9F);

  res = sc.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xA4);

  res = sc.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xA1);

  res = sc.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xE2);

  res = sc.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0x88);

  res = sc.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0x91);

  res = sc.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0);

  res = sc.popUTF8();
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(res.error(), -1);
}

TEST(LlvmLibcStringConverterTest, UTF32To8PartialRead) {
  const wchar_t *src = L"\x1f921\x2211"; // clown emoji, sigma symbol
  LIBC_NAMESPACE::internal::mbstate state;
  LIBC_NAMESPACE::internal::StringConverter<char32_t> sc(
      reinterpret_cast<const char32_t *>(src), 1, &state);

  auto res = sc.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xF0);

  res = sc.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0x9F);

  res = sc.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xA4);

  res = sc.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xA1);

  res = sc.popUTF8();
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(res.error(), -1);
}

TEST(LlvmLibcStringConverterTest, UTF8To32PartialRead) {
  // first 4 bytes are clown emoji, then next 3 are sigma symbol
  const char *src = "\xF0\x9F\xA4\xA1\xE2\x88\x91";
  LIBC_NAMESPACE::internal::mbstate state;
  LIBC_NAMESPACE::internal::StringConverter<char8_t> sc(
      reinterpret_cast<const char8_t *>(src), 5, &state);

  auto res = sc.popUTF32();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0x1f921);

  res = sc.popUTF32();
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.error()), -1);
}

TEST(LlvmLibcStringConverterTest, UTF32To8ErrorHandling) {
  const wchar_t *src = L"\x1f921\xffffff"; // clown emoji, invalid utf32
  LIBC_NAMESPACE::internal::mbstate state;
  LIBC_NAMESPACE::internal::StringConverter<char32_t> sc(
      reinterpret_cast<const char32_t *>(src), &state);

  auto res = sc.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xF0);

  res = sc.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0x9F);

  res = sc.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xA4);

  res = sc.popUTF8();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0xA1);

  res = sc.popUTF8();
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.error()), EILSEQ);
}

TEST(LlvmLibcStringConverterTest, UTF8To32ErrorHandling) {
  // first 4 bytes are clown emoji (🤡)
  // next 2 don't form a complete character
  const char *src = "\xF0\x9F\xA4\xA1\xE2\x88";
  LIBC_NAMESPACE::internal::mbstate state;
  LIBC_NAMESPACE::internal::StringConverter<char8_t> sc(
      reinterpret_cast<const char8_t *>(src), &state);

  auto res = sc.popUTF32();
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.value()), 0x1f921);

  res = sc.popUTF32();
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(static_cast<int>(res.error()), EILSEQ);
}
