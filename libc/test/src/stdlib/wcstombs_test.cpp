//===-- Unittests for wcstombs --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/wcstombs.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcWcstombs = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

// these tests are fairly simple as this function just calls into the internal
// wcsnrtombs which is more thoroughly tested

TEST_F(LlvmLibcWcstombs, AllMultibyteLengths) {
  /// clown emoji, sigma symbol, y with diaeresis, letter A
  const wchar_t src[] = {static_cast<wchar_t>(0x1f921),
                         static_cast<wchar_t>(0x2211),
                         static_cast<wchar_t>(0xff), static_cast<wchar_t>(0x41),
                         static_cast<wchar_t>(0x0)};
  char mbs[11];

  ASSERT_EQ(LIBC_NAMESPACE::wcstombs(mbs, src, 11), static_cast<size_t>(10));
  ASSERT_ERRNO_SUCCESS();
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

TEST_F(LlvmLibcWcstombs, DestLimit) {
  /// clown emoji, sigma symbol, y with diaeresis, letter A
  const wchar_t src[] = {static_cast<wchar_t>(0x1f921),
                         static_cast<wchar_t>(0x2211),
                         static_cast<wchar_t>(0xff), static_cast<wchar_t>(0x41),
                         static_cast<wchar_t>(0x0)};
  char mbs[11];
  for (int i = 0; i < 11; ++i)
    mbs[i] = '\x01'; // dummy initial values

  ASSERT_EQ(LIBC_NAMESPACE::wcstombs(mbs, src, 4), static_cast<size_t>(4));
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(mbs[0], '\xF0');
  ASSERT_EQ(mbs[1], '\x9F');
  ASSERT_EQ(mbs[2], '\xA4');
  ASSERT_EQ(mbs[3], '\xA1');
  ASSERT_EQ(mbs[4], '\x01'); // didn't write more than 4 bytes

  for (int i = 0; i < 11; ++i)
    mbs[i] = '\x01'; // dummy initial values

  // not enough bytes to convert the second character, so only converts one
  ASSERT_EQ(LIBC_NAMESPACE::wcstombs(mbs, src, 6), static_cast<size_t>(4));
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(mbs[0], '\xF0');
  ASSERT_EQ(mbs[1], '\x9F');
  ASSERT_EQ(mbs[2], '\xA4');
  ASSERT_EQ(mbs[3], '\xA1');
  ASSERT_EQ(mbs[4], '\x01');
}

TEST_F(LlvmLibcWcstombs, ErrnoTest) {
  const wchar_t src[] = {static_cast<wchar_t>(0x1f921),
                         static_cast<wchar_t>(0x2211),
                         static_cast<wchar_t>(0x12ffff), // invalid widechar
                         static_cast<wchar_t>(0x0)};
  char mbs[11];

  // n parameter ignored when dest is null
  ASSERT_EQ(LIBC_NAMESPACE::wcstombs(mbs, src, 7), static_cast<size_t>(7));
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(LIBC_NAMESPACE::wcstombs(mbs, src, 100), static_cast<size_t>(-1));
  ASSERT_ERRNO_EQ(EILSEQ);
}
