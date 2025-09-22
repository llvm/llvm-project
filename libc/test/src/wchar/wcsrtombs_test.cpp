//===-- Unittests for wcsrtombs -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/mbstate_t.h"
#include "src/__support/macros/null_check.h"
#include "src/string/memset.h"
#include "src/wchar/wcsrtombs.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcWcsrtombs = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

// these tests are fairly simple as this function just calls into the internal
// wcsnrtombs which is more thoroughly tested

TEST_F(LlvmLibcWcsrtombs, AllMultibyteLengths) {
  mbstate_t state;
  LIBC_NAMESPACE::memset(&state, 0, sizeof(mbstate_t));

  /// clown emoji, sigma symbol, y with diaeresis, letter A
  const wchar_t src[] = {static_cast<wchar_t>(0x1f921),
                         static_cast<wchar_t>(0x2211),
                         static_cast<wchar_t>(0xff), static_cast<wchar_t>(0x41),
                         static_cast<wchar_t>(0x0)};
  const wchar_t *cur = src;
  char mbs[11];

  ASSERT_EQ(LIBC_NAMESPACE::wcsrtombs(mbs, &cur, 11, &state),
            static_cast<size_t>(10));
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(cur, nullptr);
  ASSERT_EQ(mbs[0], '\xF0'); // clown begin
  ASSERT_EQ(mbs[1], '\x9F');
  ASSERT_EQ(mbs[2], '\xA4');
  ASSERT_EQ(mbs[3], '\xA1');
  ASSERT_EQ(mbs[4], '\xE2'); // sigma begin
  ASSERT_EQ(mbs[5], '\x88');
  ASSERT_EQ(mbs[6], '\x91');
  ASSERT_EQ(mbs[7], '\xC3'); // y diaeresis begin
  ASSERT_EQ(mbs[8], '\xBF');
  ASSERT_EQ(mbs[9], '\x41'); // A begin
  ASSERT_EQ(mbs[10], '\0');  // null terminator
}

TEST_F(LlvmLibcWcsrtombs, DestLimit) {
  mbstate_t state;
  LIBC_NAMESPACE::memset(&state, 0, sizeof(mbstate_t));

  /// clown emoji, sigma symbol, y with diaeresis, letter A
  const wchar_t src[] = {static_cast<wchar_t>(0x1f921),
                         static_cast<wchar_t>(0x2211),
                         static_cast<wchar_t>(0xff), static_cast<wchar_t>(0x41),
                         static_cast<wchar_t>(0x0)};
  const wchar_t *cur = src;

  char mbs[11];
  LIBC_NAMESPACE::memset(mbs, '\x01', 11); // dummy initial values

  ASSERT_EQ(LIBC_NAMESPACE::wcsrtombs(mbs, &cur, 4, &state),
            static_cast<size_t>(4));
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(cur, src + 1);
  ASSERT_EQ(mbs[0], '\xF0');
  ASSERT_EQ(mbs[1], '\x9F');
  ASSERT_EQ(mbs[2], '\xA4');
  ASSERT_EQ(mbs[3], '\xA1');
  ASSERT_EQ(mbs[4], '\x01'); // didn't write more than 4 bytes

  LIBC_NAMESPACE::memset(mbs, '\x01', 11); // dummy initial values
  LIBC_NAMESPACE::memset(&state, 0, sizeof(mbstate_t));

  // not enough bytes to convert the second character, so only converts one
  cur = src;
  ASSERT_EQ(LIBC_NAMESPACE::wcsrtombs(mbs, &cur, 6, &state),
            static_cast<size_t>(4));
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(cur, src + 1);
  ASSERT_EQ(mbs[0], '\xF0');
  ASSERT_EQ(mbs[1], '\x9F');
  ASSERT_EQ(mbs[2], '\xA4');
  ASSERT_EQ(mbs[3], '\xA1');
  ASSERT_EQ(mbs[4], '\x01');
}

TEST_F(LlvmLibcWcsrtombs, ErrnoTest) {
  mbstate_t state;
  LIBC_NAMESPACE::memset(&state, 0, sizeof(mbstate_t));

  const wchar_t src[] = {static_cast<wchar_t>(0x1f921),
                         static_cast<wchar_t>(0x2211),
                         static_cast<wchar_t>(0x12ffff), // invalid widechar
                         static_cast<wchar_t>(0x0)};
  const wchar_t *cur = src;
  char mbs[11];

  // n parameter ignored when dest is null
  ASSERT_EQ(LIBC_NAMESPACE::wcsrtombs(mbs, &cur, 7, &state),
            static_cast<size_t>(7));
  ASSERT_ERRNO_SUCCESS();

  LIBC_NAMESPACE::memset(&state, 0, sizeof(mbstate_t));
  cur = src;

  ASSERT_EQ(LIBC_NAMESPACE::wcsrtombs(mbs, &cur, 100, &state),
            static_cast<size_t>(-1));
  ASSERT_ERRNO_EQ(EILSEQ);
}

TEST_F(LlvmLibcWcsrtombs, NullState) {
  // this test is the same as DestLimit except it uses a nullptr mbstate*

  /// clown emoji, sigma symbol, y with diaeresis, letter A
  const wchar_t src[] = {static_cast<wchar_t>(0x1f921),
                         static_cast<wchar_t>(0x2211),
                         static_cast<wchar_t>(0xff), static_cast<wchar_t>(0x41),
                         static_cast<wchar_t>(0x0)};
  const wchar_t *cur = src;

  char mbs[11];
  LIBC_NAMESPACE::memset(mbs, '\x01', 11); // dummy initial values

  ASSERT_EQ(LIBC_NAMESPACE::wcsrtombs(mbs, &cur, 4, nullptr),
            static_cast<size_t>(4));
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(cur, src + 1);
  ASSERT_EQ(mbs[0], '\xF0');
  ASSERT_EQ(mbs[1], '\x9F');
  ASSERT_EQ(mbs[2], '\xA4');
  ASSERT_EQ(mbs[3], '\xA1');
  ASSERT_EQ(mbs[4], '\x01'); // didn't write more than 4 bytes

  LIBC_NAMESPACE::memset(mbs, '\x01', 11); // dummy initial values

  // not enough bytes to convert the second character, so only converts one
  cur = src;
  ASSERT_EQ(LIBC_NAMESPACE::wcsrtombs(mbs, &cur, 6, nullptr),
            static_cast<size_t>(4));
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(cur, src + 1);
  ASSERT_EQ(mbs[0], '\xF0');
  ASSERT_EQ(mbs[1], '\x9F');
  ASSERT_EQ(mbs[2], '\xA4');
  ASSERT_EQ(mbs[3], '\xA1');
  ASSERT_EQ(mbs[4], '\x01');
}
