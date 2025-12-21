//===-- Unittests for wcsnrtombs ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/null_check.h"
#include "src/__support/macros/properties/os.h"
#include "src/__support/wchar/mbstate.h"
#include "src/__support/wchar/wcsnrtombs.h"
#include "test/UnitTest/Test.h"

// TODO: add support for 16-bit widechars to remove this macro
#ifdef LIBC_TARGET_OS_IS_WINDOWS
TEST(LlvmLibcStringConverterTest, Windows) {
  // pass on windows for now
}

#else

TEST(LlvmLibcWcsnrtombs, AllMultibyteLengths) {
  LIBC_NAMESPACE::internal::mbstate state;

  /// clown emoji, sigma symbol, y with diaeresis, letter A
  const wchar_t src[] = {static_cast<wchar_t>(0x1f921),
                         static_cast<wchar_t>(0x2211),
                         static_cast<wchar_t>(0xff), static_cast<wchar_t>(0x41),
                         static_cast<wchar_t>(0x0)};
  const wchar_t *cur = src;
  char mbs[11];

  auto res = LIBC_NAMESPACE::internal::wcsnrtombs(mbs, &cur, 5, 11, &state);
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(res.value(), static_cast<size_t>(10));
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

TEST(LlvmLibcWcsnrtombs, DestLimit) {
  LIBC_NAMESPACE::internal::mbstate state1;

  /// clown emoji, sigma symbol, y with diaeresis, letter A
  const wchar_t src[] = {static_cast<wchar_t>(0x1f921),
                         static_cast<wchar_t>(0x2211),
                         static_cast<wchar_t>(0xff), static_cast<wchar_t>(0x41),
                         static_cast<wchar_t>(0x0)};
  const wchar_t *cur = src;

  char mbs[11];
  for (int i = 0; i < 11; ++i)
    mbs[i] = '\x01'; // dummy initial values

  auto res = LIBC_NAMESPACE::internal::wcsnrtombs(mbs, &cur, 5, 4, &state1);
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(res.value(), static_cast<size_t>(4));
  ASSERT_EQ(cur, src + 1);
  ASSERT_EQ(mbs[0], '\xF0');
  ASSERT_EQ(mbs[1], '\x9F');
  ASSERT_EQ(mbs[2], '\xA4');
  ASSERT_EQ(mbs[3], '\xA1');
  ASSERT_EQ(mbs[4], '\x01'); // didn't write more than 4 bytes

  for (int i = 0; i < 11; ++i)
    mbs[i] = '\x01'; // dummy initial values
  LIBC_NAMESPACE::internal::mbstate state2;

  // not enough bytes to convert the second character, so only converts one
  cur = src;
  res = LIBC_NAMESPACE::internal::wcsnrtombs(mbs, &cur, 5, 6, &state2);
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(res.value(), static_cast<size_t>(4));
  ASSERT_EQ(cur, src + 1);
  ASSERT_EQ(mbs[0], '\xF0');
  ASSERT_EQ(mbs[1], '\x9F');
  ASSERT_EQ(mbs[2], '\xA4');
  ASSERT_EQ(mbs[3], '\xA1');
  ASSERT_EQ(mbs[4], '\x01');
}

TEST(LlvmLibcWcsnrtombs, SrcLimit) {
  LIBC_NAMESPACE::internal::mbstate state;

  /// clown emoji, sigma symbol, y with diaeresis, letter A
  const wchar_t src[] = {static_cast<wchar_t>(0x1f921),
                         static_cast<wchar_t>(0x2211),
                         static_cast<wchar_t>(0xff), static_cast<wchar_t>(0x41),
                         static_cast<wchar_t>(0x0)};
  const wchar_t *cur = src;

  char mbs[11];
  for (int i = 0; i < 11; ++i)
    mbs[i] = '\x01'; // dummy initial values

  auto res = LIBC_NAMESPACE::internal::wcsnrtombs(mbs, &cur, 2, 11, &state);
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(res.value(), static_cast<size_t>(7));
  ASSERT_EQ(cur, src + 2);
  ASSERT_EQ(mbs[0], '\xF0'); // clown begin
  ASSERT_EQ(mbs[1], '\x9F');
  ASSERT_EQ(mbs[2], '\xA4');
  ASSERT_EQ(mbs[3], '\xA1');
  ASSERT_EQ(mbs[4], '\xE2'); // sigma begin
  ASSERT_EQ(mbs[5], '\x88');
  ASSERT_EQ(mbs[6], '\x91');
  ASSERT_EQ(mbs[7], '\x01');

  res = LIBC_NAMESPACE::internal::wcsnrtombs(mbs + res.value(), &cur, 100, 11,
                                             &state);
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(res.value(), static_cast<size_t>(3));
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

TEST(LlvmLibcWcsnrtombs, NullDest) {
  LIBC_NAMESPACE::internal::mbstate state1;

  const wchar_t src[] = {static_cast<wchar_t>(0x1f921),
                         static_cast<wchar_t>(0x2211),
                         static_cast<wchar_t>(0xff), static_cast<wchar_t>(0x41),
                         static_cast<wchar_t>(0x0)};
  const wchar_t *cur = src;

  // n parameter ignored when dest is null
  auto res = LIBC_NAMESPACE::internal::wcsnrtombs(nullptr, &cur, 5, 1, &state1);
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(res.value(), static_cast<size_t>(10));
  ASSERT_EQ(cur, src); // pointer not updated when dest = null

  LIBC_NAMESPACE::internal::mbstate state2;
  res = LIBC_NAMESPACE::internal::wcsnrtombs(nullptr, &cur, 5, 100, &state2);
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(res.value(), static_cast<size_t>(10));
  ASSERT_EQ(cur, src);
}

TEST(LlvmLibcWcsnrtombs, InvalidState) {
  // this is more thoroughly tested by CharacterConverter
  LIBC_NAMESPACE::internal::mbstate state;
  state.total_bytes = 100;

  const wchar_t src[] = {static_cast<wchar_t>(0x1f921),
                         static_cast<wchar_t>(0x2211),
                         static_cast<wchar_t>(0xff), static_cast<wchar_t>(0x41),
                         static_cast<wchar_t>(0x0)};
  const wchar_t *cur = src;

  // n parameter ignored when dest is null
  auto res = LIBC_NAMESPACE::internal::wcsnrtombs(nullptr, &cur, 5, 1, &state);
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(res.error(), EINVAL);
}

TEST(LlvmLibcWcsnrtombs, InvalidCharacter) {
  LIBC_NAMESPACE::internal::mbstate state1;

  const wchar_t src[] = {static_cast<wchar_t>(0x1f921),
                         static_cast<wchar_t>(0x2211),
                         static_cast<wchar_t>(0x12ffff), // invalid widechar
                         static_cast<wchar_t>(0x0)};
  const wchar_t *cur = src;
  char mbs[11];

  // n parameter ignored when dest is null
  auto res = LIBC_NAMESPACE::internal::wcsnrtombs(mbs, &cur, 5, 7, &state1);
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(res.value(), static_cast<size_t>(7));

  LIBC_NAMESPACE::internal::mbstate state2;
  cur = src;
  res = LIBC_NAMESPACE::internal::wcsnrtombs(mbs, &cur, 5, 11, &state2);
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(res.error(), EILSEQ);
}

#if defined(LIBC_ADD_NULL_CHECKS) && !defined(LIBC_HAS_SANITIZER)
TEST(LlvmLibcWcsnrtombs, NullSrc) {
  EXPECT_DEATH(
      [] {
        LIBC_NAMESPACE::internal::mbstate state;
        char mbs[10];
        LIBC_NAMESPACE::internal::wcsnrtombs(mbs, nullptr, 1, 1, &state);
      },
      WITH_SIGNAL(-1));
}
#endif // LIBC_HAS_ADDRESS_SANITIZER
#endif
