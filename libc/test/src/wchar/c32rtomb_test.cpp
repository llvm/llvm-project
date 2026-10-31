//===-- Unittests for c32rtomb --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/types/char32_t.h"
#include "hdr/types/mbstate_t.h"
#include "src/__support/wchar/mbstate.h"
#include "src/string/memset.h"
#include "src/wchar/c32rtomb.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcC32RToMBTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcC32RToMBTest, OneByte) {
  mbstate_t state;
  LIBC_NAMESPACE::memset(&state, 0, sizeof(mbstate_t));
  char32_t c32 = U'U';
  char mb[4];
  size_t cnt = LIBC_NAMESPACE::c32rtomb(mb, c32, &state);
  ASSERT_EQ(cnt, static_cast<size_t>(1));
  ASSERT_EQ(mb[0], 'U');
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcC32RToMBTest, TwoByte) {
  mbstate_t state;
  LIBC_NAMESPACE::memset(&state, 0, sizeof(mbstate_t));
  // testing utf32: 0xff -> utf8: 0xc3 0xbf
  char32_t c32 = 0xff;
  char mb[4];
  size_t cnt = LIBC_NAMESPACE::c32rtomb(mb, c32, &state);
  ASSERT_EQ(cnt, static_cast<size_t>(2));
  ASSERT_EQ(mb[0], static_cast<char>(0xc3));
  ASSERT_EQ(mb[1], static_cast<char>(0xbf));
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcC32RToMBTest, ThreeByte) {
  mbstate_t state;
  LIBC_NAMESPACE::memset(&state, 0, sizeof(mbstate_t));
  // testing utf32: 0xac15 -> utf8: 0xea 0xb0 0x95
  char32_t c32 = 0xac15;
  char mb[4];
  size_t cnt = LIBC_NAMESPACE::c32rtomb(mb, c32, &state);
  ASSERT_EQ(cnt, static_cast<size_t>(3));
  ASSERT_EQ(mb[0], static_cast<char>(0xea));
  ASSERT_EQ(mb[1], static_cast<char>(0xb0));
  ASSERT_EQ(mb[2], static_cast<char>(0x95));
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcC32RToMBTest, FourByte) {
  mbstate_t state;
  LIBC_NAMESPACE::memset(&state, 0, sizeof(mbstate_t));
  // testing utf32: 0x1f921 -> utf8: 0xf0 0x9f 0xa4 0xa1
  char32_t c32 = 0x1f921;
  char mb[4];
  size_t cnt = LIBC_NAMESPACE::c32rtomb(mb, c32, &state);
  ASSERT_EQ(cnt, static_cast<size_t>(4));
  ASSERT_EQ(mb[0], static_cast<char>(0xf0));
  ASSERT_EQ(mb[1], static_cast<char>(0x9f));
  ASSERT_EQ(mb[2], static_cast<char>(0xa4));
  ASSERT_EQ(mb[3], static_cast<char>(0xa1));
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcC32RToMBTest, NullString) {
  mbstate_t state;
  LIBC_NAMESPACE::memset(&state, 0, sizeof(mbstate_t));
  char32_t c32 = U'A';
  char mb[4];

  // should be equivalent to the call c32rtomb(buf, U'\0', state)
  size_t cnt1 = LIBC_NAMESPACE::c32rtomb(nullptr, c32, &state);
  ASSERT_ERRNO_SUCCESS();
  size_t cnt2 = LIBC_NAMESPACE::c32rtomb(mb, 0, &state);
  ASSERT_ERRNO_SUCCESS();

  ASSERT_EQ(cnt1, cnt2);
  ASSERT_EQ(cnt1, static_cast<size_t>(1));
  ASSERT_EQ(mb[0], '\0');
}

TEST_F(LlvmLibcC32RToMBTest, NullState) {
  char32_t c32 = U'A';
  char mb[4];
  size_t cnt = LIBC_NAMESPACE::c32rtomb(mb, c32, nullptr);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(cnt, static_cast<size_t>(1));
  ASSERT_EQ(mb[0], 'A');
}

TEST_F(LlvmLibcC32RToMBTest, NullWideChar) {
  mbstate_t state;
  LIBC_NAMESPACE::memset(&state, 0, sizeof(mbstate_t));
  char mb[4];
  size_t cnt = LIBC_NAMESPACE::c32rtomb(mb, 0, &state);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(cnt, static_cast<size_t>(1));
  ASSERT_EQ(mb[0], '\0');
}

TEST_F(LlvmLibcC32RToMBTest, InvalidCharTooLarge) {
  mbstate_t state;
  LIBC_NAMESPACE::memset(&state, 0, sizeof(mbstate_t));
  char32_t c32 = 0x12ffff;
  char mb[4];
  size_t cnt = LIBC_NAMESPACE::c32rtomb(mb, c32, &state);
  ASSERT_EQ(cnt, static_cast<size_t>(-1));
  ASSERT_ERRNO_EQ(EILSEQ);
}

TEST_F(LlvmLibcC32RToMBTest, InvalidSurrogate) {
  mbstate_t state;
  LIBC_NAMESPACE::memset(&state, 0, sizeof(mbstate_t));
  char mb[4];

  // Leading surrogate
  size_t cnt1 = LIBC_NAMESPACE::c32rtomb(mb, 0xd800, &state);
  ASSERT_EQ(cnt1, static_cast<size_t>(-1));
  ASSERT_ERRNO_EQ(EILSEQ);

  // Trailing surrogate
  size_t cnt2 = LIBC_NAMESPACE::c32rtomb(mb, 0xdfff, &state);
  ASSERT_EQ(cnt2, static_cast<size_t>(-1));
  ASSERT_ERRNO_EQ(EILSEQ);
}

TEST_F(LlvmLibcC32RToMBTest, InvalidMBState) {
  mbstate_t *state;
  LIBC_NAMESPACE::internal::mbstate inv;
  inv.total_bytes = 6;
  state = reinterpret_cast<mbstate_t *>(&inv);
  char32_t c32 = U'A';
  char mb[4];
  size_t cnt = LIBC_NAMESPACE::c32rtomb(mb, c32, state);
  ASSERT_EQ(cnt, static_cast<size_t>(-1));
  ASSERT_ERRNO_EQ(EINVAL);
}
