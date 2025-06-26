//===-- Unittests for wcrtomb_bounded -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/wchar/mbstate.h"
#include "src/__support/wchar/wcrtomb_bounded.h"
#include "test/UnitTest/Test.h"

// The majority of the following tests are the same as
// tests/src/wchar/wcrtomb_test.cpp

TEST(LlvmLibcWCRToMBBoundedTest, OneByte) {
  LIBC_NAMESPACE::internal::mbstate state;
  wchar_t wc = L'U';
  char mb[4];
  auto result = LIBC_NAMESPACE::internal::wcrtomb_bounded(mb, wc, &state, 4);
  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(result.value(), static_cast<size_t>(1));
  ASSERT_EQ(mb[0], 'U');
}

TEST(LlvmLibcWCRToMBBoundedTest, TwoByte) {
  LIBC_NAMESPACE::internal::mbstate state;
  // testing utf32: 0xff -> utf8: 0xc3 0xbf
  wchar_t wc = 0xff;
  char mb[4];
  auto result = LIBC_NAMESPACE::internal::wcrtomb_bounded(mb, wc, &state, 4);
  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(result.value(), static_cast<size_t>(2));
  ASSERT_EQ(mb[0], static_cast<char>(0xc3));
  ASSERT_EQ(mb[1], static_cast<char>(0xbf));
}

TEST(LlvmLibcWCRToMBBoundedTest, ThreeByte) {
  LIBC_NAMESPACE::internal::mbstate state;
  // testing utf32: 0xac15 -> utf8: 0xea 0xb0 0x95
  wchar_t wc = 0xac15;
  char mb[4];
  auto result = LIBC_NAMESPACE::internal::wcrtomb_bounded(mb, wc, &state, 4);
  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(result.value(), static_cast<size_t>(3));
  ASSERT_EQ(mb[0], static_cast<char>(0xea));
  ASSERT_EQ(mb[1], static_cast<char>(0xb0));
  ASSERT_EQ(mb[2], static_cast<char>(0x95));
}

TEST(LlvmLibcWCRToMBTest, FourByte) {
  LIBC_NAMESPACE::internal::mbstate state;
  // testing utf32: 0x1f921 -> utf8: 0xf0 0x9f 0xa4 0xa1
  wchar_t wc = 0x1f921;
  char mb[4];
  auto result = LIBC_NAMESPACE::internal::wcrtomb_bounded(mb, wc, &state, 4);
  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(result.value(), static_cast<size_t>(4));
  ASSERT_EQ(mb[0], static_cast<char>(0xf0));
  ASSERT_EQ(mb[1], static_cast<char>(0x9f));
  ASSERT_EQ(mb[2], static_cast<char>(0xa4));
  ASSERT_EQ(mb[3], static_cast<char>(0xa1));
}

TEST(LlvmLibcWCRToMBBoundedTest, NullString) {
  LIBC_NAMESPACE::internal::mbstate state;
  wchar_t wc = L'A';

  // should return the multi byte length of the widechar
  auto result =
      LIBC_NAMESPACE::internal::wcrtomb_bounded(nullptr, wc, &state, 4);

  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(result.value(), static_cast<size_t>(1));
}

TEST(LlvmLibcWCRToMBBoundedTest, InvalidWchar) {
  LIBC_NAMESPACE::internal::mbstate state;
  wchar_t wc = 0x12ffff;
  char mb[4];
  auto result = LIBC_NAMESPACE::internal::wcrtomb_bounded(mb, wc, &state, 4);
  ASSERT_FALSE(result.has_value());
  ASSERT_EQ(result.error(), EILSEQ);
}

TEST(LlvmLibcWCRToMBBoundedTest, InvalidMBState) {
  LIBC_NAMESPACE::internal::mbstate inv;
  inv.total_bytes = 6;
  wchar_t wc = L'A';
  char mb[4];
  auto result = LIBC_NAMESPACE::internal::wcrtomb_bounded(mb, wc, &inv, 4);
  ASSERT_FALSE(result.has_value());
  ASSERT_EQ(result.error(), EINVAL);
}

// wcrtomb_bounded unique tests

TEST(LlvmLibcWCRToMBBoundedTest, ContinueConversion) {
  LIBC_NAMESPACE::internal::mbstate state;
  // utf32: 0x1f921 -> utf8: 0xf0 0x9f 0xa4 0xa1
  wchar_t wc = 0x1f921;
  char mb[5] = {'\x01', '\x01', '\x01', '\x01', '\x01'};
  auto result = LIBC_NAMESPACE::internal::wcrtomb_bounded(mb, wc, &state, 1);
  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(result.value(),
            static_cast<size_t>(-1)); // conversion not completed
  ASSERT_EQ(mb[0], '\xF0');
  ASSERT_EQ(mb[1], '\x01');

  result = LIBC_NAMESPACE::internal::wcrtomb_bounded(mb + 1, wc, &state, 2);
  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(result.value(),
            static_cast<size_t>(-1)); // conversion not completed
  ASSERT_EQ(mb[0], '\xF0');
  ASSERT_EQ(mb[1], '\x9F');
  ASSERT_EQ(mb[2], '\xA4');
  ASSERT_EQ(mb[3], '\x01');

  result = LIBC_NAMESPACE::internal::wcrtomb_bounded(mb + 3, wc, &state, 100);
  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(result.value(), static_cast<size_t>(1));
  ASSERT_EQ(mb[0], '\xF0');
  ASSERT_EQ(mb[1], '\x9F');
  ASSERT_EQ(mb[2], '\xA4');
  ASSERT_EQ(mb[3], '\xA1');
  ASSERT_EQ(mb[4], '\x01');
}