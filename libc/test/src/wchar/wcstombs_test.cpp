//===-- Unittests for wcstombs --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wcstombs.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcWcstombs = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcWcstombs, AllMultibyteLengths) {
  /// clown emoji, sigma symbol, y with diaeresis, letter A
  const wchar_t src[] = {static_cast<wchar_t>(0x1f921),
                         static_cast<wchar_t>(0x2211),
                         static_cast<wchar_t>(0xff), static_cast<wchar_t>(0x41),
                         static_cast<wchar_t>(0x0)};
  char mbs[11];

  ASSERT_EQ(wcstombs(mbs, src, 11), static_cast<size_t>(11));
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

TEST_F(LlvmLibcWcstombs, PartialConversion) {
  /// clown emoji, sigma symbol, y with diaeresis, letter A
  const wchar_t src[] = {static_cast<wchar_t>(0x1f921),
                         static_cast<wchar_t>(0x2211),
                         static_cast<wchar_t>(0xff), static_cast<wchar_t>(0x41),
                         static_cast<wchar_t>(0x0)};
  char mbs[11] = {0};

  ASSERT_EQ(wcstombs(mbs, src, 6), static_cast<size_t>(4));
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(mbs[0], '\xF0'); // clown begin
  ASSERT_EQ(mbs[1], '\x9F');
  ASSERT_EQ(mbs[2], '\xA4');
  ASSERT_EQ(mbs[3], '\xA1');
  ASSERT_EQ(mbs[4], '\0');

  ASSERT_EQ(wcstombs(mbs, src, 6), static_cast<size_t>(4));

  ASSERT_EQ(mbs[4], '\xE2'); // sigma begin
  ASSERT_EQ(mbs[5], '\x88');
  ASSERT_EQ(mbs[6], '\x91');
  ASSERT_EQ(mbs[7], '\xC3'); // y diaeresis begin
  ASSERT_EQ(mbs[8], '\xBF');
  ASSERT_EQ(mbs[9], '\x41'); // A begin
  ASSERT_EQ(mbs[10], '\0');  // null terminator
}
