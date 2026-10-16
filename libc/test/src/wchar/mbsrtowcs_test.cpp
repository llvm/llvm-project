//===-- Unittests for mbsetowcs -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/types/mbstate_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/macros/null_check.h"
#include "src/__support/wchar/mbstate.h"
#include "src/string/memset.h"
#include "src/wchar/mbsrtowcs.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcMBSRToWCSTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcMBSRToWCSTest, OneByteOneChar) {
  const char *ch = "A";
  const char *original = ch;
  wchar_t dest[2];
  mbstate_t mb;
  LIBC_NAMESPACE::memset(&mb, 0, sizeof(mbstate_t));
  size_t n = LIBC_NAMESPACE::mbsrtowcs(dest, &ch, 1, &mb);
  ASSERT_EQ(static_cast<char>(*dest), 'A');
  ASSERT_EQ(static_cast<int>(n), 1);
  // Should point to null terminator now
  ASSERT_EQ(ch, original + 1);
  ASSERT_ERRNO_SUCCESS();

  n = LIBC_NAMESPACE::mbsrtowcs(dest + 1, &ch, 1, &mb);
  ASSERT_EQ(static_cast<char>(dest[1]), '\0');
  // Should not include null terminator
  ASSERT_EQ(static_cast<int>(n), 0);
  // Should now be a nullptr
  ASSERT_EQ(ch, nullptr);
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcMBSRToWCSTest, FourByteOneChar) {
  const char *src = "\xf0\x9f\x98\xb9"; // laughing cat emoji 😹
  wchar_t dest[2];
  mbstate_t mb;
  LIBC_NAMESPACE::memset(&mb, 0, sizeof(mbstate_t));
  size_t n = LIBC_NAMESPACE::mbsrtowcs(dest, &src, 2, &mb);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(static_cast<int>(dest[0]), 128569);
  ASSERT_TRUE(dest[1] == L'\0');
  // Should not count null terminator in number
  ASSERT_EQ(static_cast<int>(n), 1);
  // Should now be a nullptr
  ASSERT_EQ(src, nullptr);
}

TEST_F(LlvmLibcMBSRToWCSTest, MultiByteTwoCharacters) {
  // Two laughing cat emojis "😹😹"
  const char *src = "\xf0\x9f\x98\xb9\xf0\x9f\x98\xb9";
  wchar_t dest[3];
  mbstate_t mb;
  LIBC_NAMESPACE::memset(&mb, 0, sizeof(mbstate_t));
  size_t n = LIBC_NAMESPACE::mbsrtowcs(dest, &src, 3, &mb);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(static_cast<int>(dest[0]), 128569);
  ASSERT_EQ(static_cast<int>(dest[1]), 128569);
  ASSERT_TRUE(dest[2] == L'\0');
  // Should not count null terminator in number
  ASSERT_EQ(static_cast<int>(n), 2);
  // Should now be a nullptr
  ASSERT_EQ(src, nullptr);
}

TEST_F(LlvmLibcMBSRToWCSTest, MixedNumberOfBytes) {
  // 'A', sigma symbol 'Σ', recycling symbol '♻', laughing cat emoji '😹'
  const char *src = "A\xce\xa3\xe2\x99\xbb\xf0\x9f\x98\xb9";
  const char *original = src;
  wchar_t dest[5];
  size_t n = LIBC_NAMESPACE::mbsrtowcs(dest, &src, 4, nullptr);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(static_cast<char>(dest[0]), 'A');
  ASSERT_EQ(static_cast<int>(dest[1]), 931);
  ASSERT_EQ(static_cast<int>(dest[2]), 9851);
  ASSERT_EQ(static_cast<int>(dest[3]), 128569);
  // Should point to null terminator (byte at 10th index)
  ASSERT_EQ(src, original + 10);
  ASSERT_EQ(static_cast<int>(n), 4);
  n = LIBC_NAMESPACE::mbsrtowcs(dest + 4, &src, 4, nullptr);
  ASSERT_TRUE(dest[4] == L'\0');
  ASSERT_ERRNO_SUCCESS();
  // Should not count null terminator in number
  ASSERT_EQ(static_cast<int>(n), 0);
  // Should now be a nullptr
  ASSERT_EQ(src, nullptr);
}

TEST_F(LlvmLibcMBSRToWCSTest, ReadLessThanStringLength) {
  // Four laughing cat emojis "😹😹😹😹"
  const char *src =
      "\xf0\x9f\x98\xb9\xf0\x9f\x98\xb9\xf0\x9f\x98\xb9\xf0\x9f\x98\xb9";
  const char *original = src;
  wchar_t dest[5] = {L'a', L'b', L'c', L'd', L'e'};
  size_t n = LIBC_NAMESPACE::mbsrtowcs(dest, &src, 3, nullptr);
  ASSERT_ERRNO_SUCCESS();
  // Should have read 3 emojis
  ASSERT_EQ(static_cast<int>(n), 3);
  ASSERT_EQ(static_cast<int>(dest[0]), 128569);
  ASSERT_EQ(static_cast<int>(dest[1]), 128569);
  ASSERT_EQ(static_cast<int>(dest[2]), 128569);
  ASSERT_TRUE(dest[3] == L'd');
  ASSERT_TRUE(dest[4] == L'e');
  // Read three laughing cat emojis, 12 bytes
  ASSERT_EQ(src, original + 12);
}

TEST_F(LlvmLibcMBSRToWCSTest, InvalidFirstByte) {
  // 0x80 is invalid first byte of mb character
  const char *src =
      "\x80\x9f\x98\xb9\xf0\x9f\x98\xb9\xf0\x9f\x98\xb9\xf0\x9f\x98\xb9";
  const char *original = src;
  wchar_t dest[3];
  size_t n = LIBC_NAMESPACE::mbsrtowcs(dest, &src, 3, nullptr);
  // Should return error and set errno
  ASSERT_EQ(static_cast<int>(n), -1);
  ASSERT_ERRNO_EQ(EILSEQ);
  // Should not update pointer
  ASSERT_EQ(src, original);
}

TEST_F(LlvmLibcMBSRToWCSTest, InvalidMiddleByte) {
  // The 7th byte is invalid for a 4 byte character
  const char *src =
      "\xf0\x9f\x98\xb9\xf0\x9f\xf0\xb9\xf0\x9f\x98\xb9\xf0\x9f\x98\xb9";
  const char *original = src;
  wchar_t dest[3];
  size_t n = LIBC_NAMESPACE::mbsrtowcs(dest, &src, 5, nullptr);
  // Should return error, set errno, and not update the pointer
  ASSERT_EQ(static_cast<int>(n), -1);
  ASSERT_ERRNO_EQ(EILSEQ);
  ASSERT_EQ(src, original);
}

TEST_F(LlvmLibcMBSRToWCSTest, NullDestination) {
  // Four laughing cat emojis "😹😹😹😹"
  const char *src =
      "\xf0\x9f\x98\xb9\xf0\x9f\x98\xb9\xf0\x9f\x98\xb9\xf0\x9f\x98\xb9";
  const char *original = src;
  size_t n = LIBC_NAMESPACE::mbsrtowcs(nullptr, &src, 2, nullptr);
  ASSERT_ERRNO_SUCCESS();
  // Null destination should ignore len and read till end of string
  ASSERT_EQ(static_cast<int>(n), 4);
  // It should also not change the src pointer
  ASSERT_EQ(src, original);
}

TEST_F(LlvmLibcMBSRToWCSTest, ErrnoChecks) {
  // Two laughing cat emojis and invalid 3rd mb char (3rd byte of it)
  const char *src =
      "\xf0\x9f\x98\xb9\xf0\x9f\x98\xb9\xf0\x9f\xf0\xb9\xf0\x9f\x98\xb9";
  const char *original = src;
  wchar_t dest[5];
  // First two bytes are valid --> should not set errno
  size_t n = LIBC_NAMESPACE::mbsrtowcs(dest, &src, 2, nullptr);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(static_cast<int>(n), 2);
  ASSERT_EQ(static_cast<int>(dest[0]), 128569);
  ASSERT_EQ(static_cast<int>(dest[1]), 128569);
  ASSERT_EQ(src, original + 8);

  // Trying to read the 3rd byte should set errno
  n = LIBC_NAMESPACE::mbsrtowcs(dest, &src, 2, nullptr);
  ASSERT_ERRNO_EQ(EILSEQ);
  ASSERT_EQ(static_cast<int>(n), -1);
  // Should not move the pointer
  ASSERT_EQ(src, original + 8);
}

#if defined(LIBC_ADD_NULL_CHECKS)
TEST(LlvmLibcMBSRToWCSTest, NullptrCrash) {
  // Passing in a nullptr should crash the program.
  EXPECT_DEATH([] { LIBC_NAMESPACE::mbsrtowcs(nullptr, nullptr, 1, nullptr); },
               WITH_SIGNAL(-1));
}
#endif // LIBC_ADD_NULL_CHECKS
