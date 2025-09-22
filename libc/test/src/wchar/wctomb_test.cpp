//===-- Unittests for wctomb ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/types/wchar_t.h"
#include "src/wchar/wctomb.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcWCToMBTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST(LlvmLibcWCToMBTest, OneByte) {
  wchar_t wc = L'U';
  char mb[4];
  int cnt = LIBC_NAMESPACE::wctomb(mb, wc);
  ASSERT_EQ(cnt, 1);
  ASSERT_EQ(mb[0], 'U');
}

TEST(LlvmLibcWCToMBTest, TwoByte) {
  // testing utf32: 0xff -> utf8: 0xc3 0xbf
  wchar_t wc = 0xff;
  char mb[4];
  int cnt = LIBC_NAMESPACE::wctomb(mb, wc);
  ASSERT_EQ(cnt, 2);
  ASSERT_EQ(mb[0], static_cast<char>(0xc3));
  ASSERT_EQ(mb[1], static_cast<char>(0xbf));
}

TEST(LlvmLibcWCToMBTest, ThreeByte) {
  // testing utf32: 0xac15 -> utf8: 0xea 0xb0 0x95
  wchar_t wc = 0xac15;
  char mb[4];
  int cnt = LIBC_NAMESPACE::wctomb(mb, wc);
  ASSERT_EQ(cnt, 3);
  ASSERT_EQ(mb[0], static_cast<char>(0xea));
  ASSERT_EQ(mb[1], static_cast<char>(0xb0));
  ASSERT_EQ(mb[2], static_cast<char>(0x95));
}

TEST(LlvmLibcWCToMBTest, FourByte) {
  // testing utf32: 0x1f921 -> utf8: 0xf0 0x9f 0xa4 0xa1
  wchar_t wc = 0x1f921;
  char mb[4];
  int cnt = LIBC_NAMESPACE::wctomb(mb, wc);
  ASSERT_EQ(cnt, 4);
  ASSERT_EQ(mb[0], static_cast<char>(0xf0));
  ASSERT_EQ(mb[1], static_cast<char>(0x9f));
  ASSERT_EQ(mb[2], static_cast<char>(0xa4));
  ASSERT_EQ(mb[3], static_cast<char>(0xa1));
}

TEST(LlvmLibcWCToMBTest, NullString) {
  wchar_t wc = L'A';

  int cnt = LIBC_NAMESPACE::wctomb(nullptr, wc);

  // no state-dependent encoding
  ASSERT_EQ(cnt, 0);
}

TEST(LlvmLibcWCToMBTest, InvalidWchar) {
  wchar_t wc = 0x12ffff;
  char mb[4];
  int cnt = LIBC_NAMESPACE::wctomb(mb, wc);
  ASSERT_EQ(cnt, -1);
  ASSERT_ERRNO_EQ(EILSEQ);
}
