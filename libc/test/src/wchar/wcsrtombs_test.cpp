//===-- Unittests for wcsrtombs ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/mbstate_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/wchar/mbstate.h"
#include "src/string/memset.h"
#include "src/wchar/wcsrtombs.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcWCSRToMBSTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcWCSRToMBSTest, SingleCharacterOneByte) {
  mbstate_t state;
  LIBC_NAMESPACE::memset(&state, 0, sizeof(mbstate_t));
  const wchar_t *wcs = L"U";
  char mbs[] = {0, 0};
  size_t cnt = LIBC_NAMESPACE::wcsrtombs(mbs, &wcs, 2, &state);
  ASSERT_EQ(cnt, static_cast<size_t>(1));
  ASSERT_EQ(mbs[0], 'U');
  ASSERT_EQ(mbs[1], '\0');
  ASSERT_EQ(wcs, nullptr);
}

TEST_F(LlvmLibcWCSRToMBSTest, MultipleCompleteConversions) {
  mbstate_t state;
  LIBC_NAMESPACE::memset(&state, 0, sizeof(mbstate_t));

  // utf32: 0xff -> utf8: 0xc3 0xbf
  // utf32: 0xac15 -> utf8: 0xea 0xb0 0x95
  const wchar_t *wcs = L"\xFF\xAC15";
  const wchar_t *wcs_start = wcs;

  // init with dummy value of 1 so that we can check when null byte written
  char mbs[7] = {1, 1, 1, 1, 1, 1, 1};
  char expected[6] = {'\xC3', '\xBF', '\xEA', '\xB0', '\x95', '\x00'};

  size_t cnt1 = LIBC_NAMESPACE::wcsrtombs(mbs, &wcs, 2, &state);
  ASSERT_EQ(cnt1, static_cast<size_t>(2));
  ASSERT_EQ(wcs, wcs_start + 1);
  ASSERT_EQ(mbs[0], expected[0]);
  ASSERT_EQ(mbs[1], expected[1]);
  ASSERT_EQ(mbs[2], '\x01'); // not modified

  size_t cnt2 = LIBC_NAMESPACE::wcsrtombs(mbs + cnt1, &wcs, 3, &state);
  ASSERT_EQ(cnt2, static_cast<size_t>(3));
  ASSERT_EQ(wcs, wcs_start + 2);
  ASSERT_EQ(mbs[0], expected[0]);
  ASSERT_EQ(mbs[1], expected[1]);
  ASSERT_EQ(mbs[2], expected[2]);
  ASSERT_EQ(mbs[3], expected[3]);
  ASSERT_EQ(mbs[4], expected[4]);
  ASSERT_EQ(mbs[5], '\x01'); // null byte not yet written

  // all that is left in the string is the null terminator
  size_t cnt3 = LIBC_NAMESPACE::wcsrtombs(mbs + cnt1 + cnt2, &wcs, 50, &state);
  ASSERT_EQ(cnt3, static_cast<size_t>(0));
  ASSERT_EQ(wcs, nullptr);
  ASSERT_EQ(mbs[0], expected[0]);
  ASSERT_EQ(mbs[1], expected[1]);
  ASSERT_EQ(mbs[2], expected[2]);
  ASSERT_EQ(mbs[3], expected[3]);
  ASSERT_EQ(mbs[4], expected[4]);
  ASSERT_EQ(mbs[5], expected[5]);
  ASSERT_EQ(mbs[6], '\x01'); // should not write beyond null terminator
}

TEST_F(LlvmLibcWCSRToMBSTest, MultiplePartialConversions) {
  mbstate_t state;
  LIBC_NAMESPACE::memset(&state, 0, sizeof(mbstate_t));

  // utf32: 0xff -> utf8: 0xc3 0xbf
  // utf32: 0xac15 -> utf8: 0xea 0xb0 0x95
  const wchar_t *wcs = L"\xFF\xAC15";
  const wchar_t *wcs_start = wcs;

  // init with dummy value of 1 so that we can check when null byte written
  char mbs[7] = {1, 1, 1, 1, 1, 1, 1};
  char expected[6] = {'\xC3', '\xBF', '\xEA', '\xB0', '\x95', '\x00'};
  size_t written = 0;
  size_t count = 0;

  count = LIBC_NAMESPACE::wcsrtombs(mbs, &wcs, 1, &state);
  written += count;
  // ASSERT_EQ(count, static_cast<size_t>(1));
  ASSERT_EQ(wcs, wcs_start);
  ASSERT_EQ(mbs[0], expected[0]);
  ASSERT_EQ(mbs[1], '\x01');

  count = LIBC_NAMESPACE::wcsrtombs(mbs + written, &wcs, 2, &state);
  written += count;
  ASSERT_EQ(count, static_cast<size_t>(2));
  ASSERT_EQ(wcs, wcs_start + 1);
  ASSERT_EQ(mbs[0], expected[0]);
  ASSERT_EQ(mbs[1], expected[1]);
  ASSERT_EQ(mbs[2], expected[2]);
  ASSERT_EQ(mbs[3], '\x01');

  count = LIBC_NAMESPACE::wcsrtombs(mbs + written, &wcs, 3, &state);
  written += count;
  ASSERT_EQ(count, static_cast<size_t>(2));
  ASSERT_EQ(wcs, nullptr);
  ASSERT_EQ(mbs[0], expected[0]);
  ASSERT_EQ(mbs[1], expected[1]);
  ASSERT_EQ(mbs[2], expected[2]);
  ASSERT_EQ(mbs[3], expected[3]);
  ASSERT_EQ(mbs[4], expected[4]);
  ASSERT_EQ(mbs[5], expected[5]);
  ASSERT_EQ(mbs[6], '\x01');
}

TEST_F(LlvmLibcWCSRToMBSTest, NullDestination) {
  mbstate_t state;
  LIBC_NAMESPACE::memset(&state, 0, sizeof(mbstate_t));

  // utf32: 0x1f921 -> utf8: 0xf0 0x9f 0xa4 0xa1
  // utf32: 0xac15 -> utf8: 0xea 0xb0 0x95
  const wchar_t *wcs = L"\x1F921\xAC15";

  // null destination means the conversion isnt stored, but all the side effects
  // still occur. the len parameter is also ignored
  size_t count = LIBC_NAMESPACE::wcsrtombs(nullptr, &wcs, 3, &state);
  ASSERT_EQ(count, static_cast<size_t>(7));
  ASSERT_EQ(wcs, nullptr);
}

TEST_F(LlvmLibcWCSRToMBSTest, NullState) {
  // same as MultiplePartialConversions test except without an explicit
  // mbstate_t

  const wchar_t *wcs = L"\xFF\xAC15";
  const wchar_t *wcs_start = wcs;

  // init with dummy value of 1 so that we can check when null byte written
  char mbs[7] = {1, 1, 1, 1, 1, 1, 1};
  char expected[6] = {'\xC3', '\xBF', '\xEA', '\xB0', '\x95', '\x00'};
  size_t written = 0;
  size_t count = 0;

  count = LIBC_NAMESPACE::wcsrtombs(mbs, &wcs, 1, nullptr);
  written += count;
  ASSERT_EQ(count, static_cast<size_t>(1));
  ASSERT_EQ(wcs, wcs_start);
  ASSERT_EQ(mbs[0], expected[0]);
  ASSERT_EQ(mbs[1], '\x01');

  count = LIBC_NAMESPACE::wcsrtombs(mbs + written, &wcs, 2, nullptr);
  written += count;
  ASSERT_EQ(count, static_cast<size_t>(2));
  ASSERT_EQ(wcs, wcs_start + 1);
  ASSERT_EQ(mbs[0], expected[0]);
  ASSERT_EQ(mbs[1], expected[1]);
  ASSERT_EQ(mbs[2], expected[2]);
  ASSERT_EQ(mbs[3], '\x01');

  count = LIBC_NAMESPACE::wcsrtombs(mbs + written, &wcs, 3, nullptr);
  written += count;
  ASSERT_EQ(count, static_cast<size_t>(2));
  ASSERT_EQ(wcs, nullptr);
  ASSERT_EQ(mbs[0], expected[0]);
  ASSERT_EQ(mbs[1], expected[1]);
  ASSERT_EQ(mbs[2], expected[2]);
  ASSERT_EQ(mbs[3], expected[3]);
  ASSERT_EQ(mbs[4], expected[4]);
  ASSERT_EQ(mbs[5], expected[5]);
  ASSERT_EQ(mbs[6], '\x01');
}

TEST_F(LlvmLibcWCSRToMBSTest, InvalidWchar) {
  mbstate_t state;
  LIBC_NAMESPACE::memset(&state, 0, sizeof(mbstate_t));

  const wchar_t *wcs = L"\xFF\xAC15\x12FFFF";
  char mbs[15];
  // convert the valid wchar
  size_t count = LIBC_NAMESPACE::wcsrtombs(mbs, &wcs, 5, &state);
  ASSERT_EQ(count, static_cast<size_t>(5));
  ASSERT_TRUE(*wcs == static_cast<wchar_t>(0x12ffff));
  ASSERT_ERRNO_SUCCESS();

  count = LIBC_NAMESPACE::wcsrtombs(mbs + count, &wcs, 5, &state); // invalid
  ASSERT_EQ(count, static_cast<size_t>(-1));
  ASSERT_ERRNO_EQ(EILSEQ);
}

TEST_F(LlvmLibcWCSRToMBSTest, InvalidState) {
  LIBC_NAMESPACE::internal::mbstate state{0, 0, 9}; // 9 total bytes is invalid
  const wchar_t *wcs = L"\xFF\xAC15";
  char mbs[5];
  // convert the valid wchar
  size_t count = LIBC_NAMESPACE::wcsrtombs(
      mbs, &wcs, 5, reinterpret_cast<mbstate_t *>(&state));
  ASSERT_EQ(count, static_cast<size_t>(-1));
  ASSERT_ERRNO_EQ(EINVAL);
}

#if defined(LIBC_ADD_NULL_CHECKS) && !defined(LIBC_HAS_SANITIZER)
TEST_F(LlvmLibcWCSRToMBSTest, NullSrc) {
  // Passing in a nullptr should crash the program.
  char mbs[] = {0, 0};
  EXPECT_DEATH(
      [&mbs] {
        LIBC_NAMESPACE::wcsrtombs(mbs, nullptr, 2, nullptr);
      },
      WITH_SIGNAL(-1));
}
#endif // LIBC_HAS_ADDRESS_SANITIZER
