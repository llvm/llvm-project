//===-- Unittests for mbsrtowcs -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/wchar_t.h"
#include "src/__support/libc_errno.h"
#include "src/__support/wchar/mbstate.h"
#include "src/string/memset.h"
#include "src/wchar/mbsrtowcs.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcMBSRToWCSTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcMBSRToWCSTest, OneByteOneCharacter) {
  mbstate_t *mb;
  LIBC_NAMESPACE::memset(&mb, 0, sizeof(mbstate_t));
  const char *ch = "A";
  wchar_t dest[2];
  size_t n = LIBC_NAMESPACE::mbsrtowcs(dest, &ch, 2, mb);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_TRUE(dest[0] == L'A');
  ASSERT_TRUE(dest[1] == L'\0');
  // Should not count null terminator in number
  ASSERT_EQ(static_cast<int>(n), 1);
  // Should set ch to nullptr after reading null terminator
  ASSERT_EQ(ch, nullptr);
}

TEST_F(LlvmLibcMBSRToWCSTest, MultiByteOneCharacter) {
  const char *src = "\xf0\x9f\x98\xb9"; // laughing cat emoji 😹
  wchar_t dest[2];
  size_t n = LIBC_NAMESPACE::mbsrtowcs(dest, &src, 2, nullptr);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(static_cast<int>(dest[0]), 128569);
  ASSERT_TRUE(dest[1] == L'\0');
  // Should not count null terminator in number
  ASSERT_EQ(static_cast<int>(n), 1);
  // Should set ch to nullptr after reading null terminator
  ASSERT_EQ(src, nullptr);
}

TEST_F(LlvmLibcMBSRToWCSTest, MultiByteTwoCharacters) {
  // Two laughing cat emojis "😹😹"
  const char *src = "\xf0\x9f\x98\xb9\xf0\x9f\x98\xb9";
  wchar_t dest[3];
  size_t n = LIBC_NAMESPACE::mbsrtowcs(dest, &src, 3, nullptr);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(static_cast<int>(dest[0]), 128569);
  ASSERT_EQ(static_cast<int>(dest[1]), 128569);
  ASSERT_TRUE(dest[2] == L'\0');
  // Should not count null terminator in number
  ASSERT_EQ(static_cast<int>(n), 2);
  // Should set ch to nullptr after reading null terminator
  ASSERT_EQ(src, nullptr);
}

TEST_F(LlvmLibcMBSRToWCSTest, ReadLessThanStringLength) {
  // Four laughing cat emojis "😹😹😹😹"
  const char *src =
      "\xf0\x9f\x98\xb9\xf0\x9f\x98\xb9\xf0\x9f\x98\xb9\xf0\x9f\x98\xb9";
  const char *check = src;
  wchar_t dest[3];
  size_t n = LIBC_NAMESPACE::mbsrtowcs(dest, &src, 3, nullptr);
  ASSERT_ERRNO_SUCCESS();
  // Should have read 3 emojis
  ASSERT_EQ(static_cast<int>(n), 3);
  ASSERT_EQ(static_cast<int>(dest[0]), 128569);
  ASSERT_EQ(static_cast<int>(dest[1]), 128569);
  ASSERT_EQ(static_cast<int>(dest[2]), 128569);
  // src should now point to the 4th cat emoji aka 13th byte
  ASSERT_EQ((check + 12), src);
}

TEST_F(LlvmLibcMBSRToWCSTest, InvalidFirstByte) {
  // 0x80 is invalid first byte of mb character
  const char *src =
      "\x80\x9f\x98\xb9\xf0\x9f\x98\xb9\xf0\x9f\x98\xb9\xf0\x9f\x98\xb9";
  wchar_t dest[3];
  size_t n = LIBC_NAMESPACE::mbsrtowcs(dest, &src, 3, nullptr);
  // Should return error and set errno
  ASSERT_EQ(static_cast<int>(n), -1);
  ASSERT_ERRNO_EQ(EILSEQ);
}

TEST_F(LlvmLibcMBSRToWCSTest, InvalidMiddleByte) {
  // The 7th byte is invalid for a 4 byte character
  const char *src =
      "\xf0\x9f\x98\xb9\xf0\x9f\xf0\xb9\xf0\x9f\x98\xb9\xf0\x9f\x98\xb9";
  wchar_t dest[3];
  size_t n = LIBC_NAMESPACE::mbsrtowcs(dest, &src, 5, nullptr);
  // Should return error and set errno
  ASSERT_EQ(static_cast<int>(n), -1);
  ASSERT_ERRNO_EQ(EILSEQ);
}

TEST_F(LlvmLibcMBSRToWCSTest, NullDestination) {
  // Four laughing cat emojis "😹😹😹😹"
  const char *src =
      "\xf0\x9f\x98\xb9\xf0\x9f\x98\xb9\xf0\x9f\x98\xb9\xf0\x9f\x98\xb9";
  size_t n = LIBC_NAMESPACE::mbsrtowcs(nullptr, &src, 5, nullptr);
  ASSERT_ERRNO_SUCCESS();
  // Null destination should still return correct number of read chars
  ASSERT_EQ(static_cast<int>(n), 4);
}

TEST_F(LlvmLibcMBSRToWCSTest, InvalidMBState) {
  mbstate_t *mb;
  LIBC_NAMESPACE::internal::mbstate inv;
  inv.total_bytes = 6;
  mb = reinterpret_cast<mbstate_t *>(&inv);
  // Four laughing cat emojis "😹😹😹😹"
  const char *src =
      "\xf0\x9f\x98\xb9\xf0\x9f\x98\xb9\xf0\x9f\x98\xb9\xf0\x9f\x98\xb9";
  wchar_t dest[3];
  size_t n = LIBC_NAMESPACE::mbsrtowcs(dest, &src, 3, mb);
  // Should fail from invalid mbstate
  ASSERT_EQ(static_cast<int>(n), -1);
  ASSERT_ERRNO_EQ(EINVAL);
}

#if defined(LIBC_ADD_NULL_CHECKS) && !defined(LIBC_HAS_SANITIZER)
TEST_F(LlvmLibcMBSRToWCSTest, NullSource) {
  // Passing in a nullptr source should crash the program
  EXPECT_DEATH([] { LIBC_NAMESPACE::mbsrtowcs(nullptr, nullptr, 1, nullptr); },
               WITH_SIGNAL(-1));
}
#endif // LIBC_HAS_ADDRESS_SANITIZER
