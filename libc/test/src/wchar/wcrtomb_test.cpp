//===-- Unittests for wcrtomb --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/mbstate_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/libc_errno.h"
#include "src/string/memset.h"
#include "src/wchar/wcrtomb.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcWCRToMBTest, OneByte) {
  mbstate_t state;
  LIBC_NAMESPACE::memset(&state, 0, sizeof(mbstate_t));
  wchar_t wc = L'U';
  char mb[4];
  size_t cnt = LIBC_NAMESPACE::wcrtomb(mb, wc, &state);
  ASSERT_EQ(cnt, static_cast<size_t>(1));
  ASSERT_EQ(mb[0], 'U');
}

TEST(LlvmLibcWCRToMBTest, TwoByte) {
  mbstate_t state;
  LIBC_NAMESPACE::memset(&state, 0, sizeof(mbstate_t));
  // testing utf32: 0xff -> utf8: 0xc3 0xbf
  wchar_t wc = 0xff;
  char mb[4];
  size_t cnt = LIBC_NAMESPACE::wcrtomb(mb, wc, &state);
  ASSERT_EQ(cnt, static_cast<size_t>(2));
  ASSERT_EQ(mb[0], static_cast<char>(0xc3));
  ASSERT_EQ(mb[1], static_cast<char>(0xbf));
}

TEST(LlvmLibcWCRToMBTest, ThreeByte) {
  mbstate_t state;
  LIBC_NAMESPACE::memset(&state, 0, sizeof(mbstate_t));
  // testing utf32: 0xac15 -> utf8: 0xea 0xb0 0x95
  wchar_t wc = 0xac15;
  char mb[4];
  size_t cnt = LIBC_NAMESPACE::wcrtomb(mb, wc, &state);
  ASSERT_EQ(cnt, static_cast<size_t>(3));
  ASSERT_EQ(mb[0], static_cast<char>(0xea));
  ASSERT_EQ(mb[1], static_cast<char>(0xb0));
  ASSERT_EQ(mb[2], static_cast<char>(0x95));
}

TEST(LlvmLibcWCRToMBTest, FourByte) {
  mbstate_t state;
  LIBC_NAMESPACE::memset(&state, 0, sizeof(mbstate_t));
  // testing utf32: 0x1f921 -> utf8: 0xf0 0x9f 0xa4 0xa1
  wchar_t wc = 0x1f921;
  char mb[4];
  size_t cnt = LIBC_NAMESPACE::wcrtomb(mb, wc, &state);
  ASSERT_EQ(cnt, static_cast<size_t>(4));
  ASSERT_EQ(mb[0], static_cast<char>(0xf0));
  ASSERT_EQ(mb[1], static_cast<char>(0x9f));
  ASSERT_EQ(mb[2], static_cast<char>(0xa4));
  ASSERT_EQ(mb[3], static_cast<char>(0xa1));
}

TEST(LlvmLibcWCRToMBTest, NullString) {
  mbstate_t state;
  LIBC_NAMESPACE::memset(&state, 0, sizeof(mbstate_t));
  wchar_t wc = L'A';
  char mb[4];

  // should be equivalent to the call wcrtomb(buf, L'\0', state)
  size_t cnt1 = LIBC_NAMESPACE::wcrtomb(nullptr, wc, &state);
  size_t cnt2 = LIBC_NAMESPACE::wcrtomb(mb, L'\0', &state);

  ASSERT_EQ(cnt1, cnt2);
}

TEST(LlvmLibcWCRToMBTest, NullState) {
  wchar_t wc = L'A';
  char mb[4];
  size_t cnt = LIBC_NAMESPACE::wcrtomb(mb, wc, nullptr);
  ASSERT_EQ(cnt, static_cast<size_t>(1));
}

TEST(LlvmLibcWCRToMBTest, InvalidWchar) {
  mbstate_t state;
  LIBC_NAMESPACE::memset(&state, 0, sizeof(mbstate_t));
  wchar_t wc = 0x12ffff;
  char mb[4];
  size_t cnt = LIBC_NAMESPACE::wcrtomb(mb, wc, &state);
  ASSERT_EQ(cnt, static_cast<size_t>(-1));
  ASSERT_EQ(static_cast<int>(libc_errno), EILSEQ);
}
