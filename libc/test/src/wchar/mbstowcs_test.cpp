//===-- Unittests for mbstowcs --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/macros/null_check.h"
#include "src/wchar/mbstowcs.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcMBSToWCSTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcMBSToWCSTest, OneByteOneChar) {
  const char *ch = "A";
  const char *original = ch;
  wchar_t dest[2];
  size_t n = LIBC_NAMESPACE::mbstowcs(dest, ch, 1);
  ASSERT_EQ(static_cast<char>(*dest), 'A');
  ASSERT_EQ(static_cast<int>(n), 1);
  // Making sure the pointer is not getting updated
  ASSERT_EQ(ch, original);
  ASSERT_ERRNO_SUCCESS();

  n = LIBC_NAMESPACE::mbstowcs(dest + 1, ch + 1, 1);
  ASSERT_EQ(static_cast<char>(dest[1]), '\0');
  // Should not include null terminator
  ASSERT_EQ(static_cast<int>(n), 0);
  // Making sure the pointer is not getting updated
  ASSERT_EQ(ch, original);
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcMBSToWCSTest, FourByteOneChar) {
  const char *src = "\xf0\x9f\x98\xb9"; // laughing cat emoji 😹
  const char *original = src;
  wchar_t dest[2];
  size_t n = LIBC_NAMESPACE::mbstowcs(dest, src, 2);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(static_cast<int>(dest[0]), 128569);
  ASSERT_TRUE(dest[1] == L'\0');
  // Should not count null terminator in number
  ASSERT_EQ(static_cast<int>(n), 1);
  // Making sure the pointer is not getting updated
  ASSERT_EQ(src, original);
}

TEST_F(LlvmLibcMBSToWCSTest, MultiByteTwoCharacters) {
  // Two laughing cat emojis "😹😹"
  const char *src = "\xf0\x9f\x98\xb9\xf0\x9f\x98\xb9";
  const char *original = src;
  wchar_t dest[3];
  size_t n = LIBC_NAMESPACE::mbstowcs(dest, src, 3);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(static_cast<int>(dest[0]), 128569);
  ASSERT_EQ(static_cast<int>(dest[1]), 128569);
  ASSERT_TRUE(dest[2] == L'\0');
  // Should not count null terminator in number
  ASSERT_EQ(static_cast<int>(n), 2);
  // Making sure the pointer is not getting updated
  ASSERT_EQ(src, original);
}

TEST_F(LlvmLibcMBSToWCSTest, MixedNumberOfBytes) {
  // 'A', sigma symbol 'Σ', recycling symbol '♻', laughing cat emoji '😹'
  const char *src = "A\xce\xa3\xe2\x99\xbb\xf0\x9f\x98\xb9";
  const char *original = src;
  wchar_t dest[5];
  size_t n = LIBC_NAMESPACE::mbstowcs(dest, src, 5);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(static_cast<char>(dest[0]), 'A');
  ASSERT_EQ(static_cast<int>(dest[1]), 931);
  ASSERT_EQ(static_cast<int>(dest[2]), 9851);
  ASSERT_EQ(static_cast<int>(dest[3]), 128569);
  ASSERT_TRUE(dest[4] == L'\0');
  // Should not count null terminator in number
  ASSERT_EQ(static_cast<int>(n), 4);
  // Making sure the pointer is not getting updated
  ASSERT_EQ(src, original);
}

TEST_F(LlvmLibcMBSToWCSTest, ReadLessThanStringLength) {
  // Four laughing cat emojis "😹😹😹😹"
  const char *src =
      "\xf0\x9f\x98\xb9\xf0\x9f\x98\xb9\xf0\x9f\x98\xb9\xf0\x9f\x98\xb9";
  const char *original = src;
  wchar_t dest[5] = {L'a', L'b', L'c', L'd', L'e'};
  size_t n = LIBC_NAMESPACE::mbstowcs(dest, src, 3);
  ASSERT_ERRNO_SUCCESS();
  // Should have read 3 emojis
  ASSERT_EQ(static_cast<int>(n), 3);
  ASSERT_EQ(static_cast<int>(dest[0]), 128569);
  ASSERT_EQ(static_cast<int>(dest[1]), 128569);
  ASSERT_EQ(static_cast<int>(dest[2]), 128569);
  ASSERT_TRUE(dest[3] == L'd');
  ASSERT_TRUE(dest[4] == L'e');
  // Making sure the pointer is not getting updated
  ASSERT_EQ(src, original);
}

TEST_F(LlvmLibcMBSToWCSTest, InvalidFirstByte) {
  // 0x80 is invalid first byte of mb character
  const char *src =
      "\x80\x9f\x98\xb9\xf0\x9f\x98\xb9\xf0\x9f\x98\xb9\xf0\x9f\x98\xb9";
  wchar_t dest[3];
  size_t n = LIBC_NAMESPACE::mbstowcs(dest, src, 3);
  // Should return error and set errno
  ASSERT_EQ(static_cast<int>(n), -1);
  ASSERT_ERRNO_EQ(EILSEQ);
}

TEST_F(LlvmLibcMBSToWCSTest, InvalidMiddleByte) {
  // The 7th byte is invalid for a 4 byte character
  const char *src =
      "\xf0\x9f\x98\xb9\xf0\x9f\xf0\xb9\xf0\x9f\x98\xb9\xf0\x9f\x98\xb9";
  const char *original = src;
  wchar_t dest[3];
  size_t n = LIBC_NAMESPACE::mbstowcs(dest, src, 5);
  // Should return error and set errno
  ASSERT_EQ(static_cast<int>(n), -1);
  ASSERT_ERRNO_EQ(EILSEQ);
  // Making sure the pointer is not getting updated
  ASSERT_EQ(src, original);
}

TEST_F(LlvmLibcMBSToWCSTest, NullDestination) {
  // Four laughing cat emojis "😹😹😹😹"
  const char *src =
      "\xf0\x9f\x98\xb9\xf0\x9f\x98\xb9\xf0\x9f\x98\xb9\xf0\x9f\x98\xb9";
  const char *original = src;
  size_t n = LIBC_NAMESPACE::mbstowcs(nullptr, src, 2);
  ASSERT_ERRNO_SUCCESS();
  // Null destination should ignore len and read till end of string
  ASSERT_EQ(static_cast<int>(n), 4);
  // Making sure the pointer is not getting updated
  ASSERT_EQ(src, original);
}

TEST_F(LlvmLibcMBSToWCSTest, ErrnoChecks) {
  // Two laughing cat emojis and invalid 3rd mb char (3rd byte of it)
  const char *src =
      "\xf0\x9f\x98\xb9\xf0\x9f\x98\xb9\xf0\x9f\xf0\xb9\xf0\x9f\x98\xb9";
  const char *original = src;
  wchar_t dest[5];
  // First two bytes are valid --> should not set errno
  size_t n = LIBC_NAMESPACE::mbstowcs(dest, src, 2);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(static_cast<int>(n), 2);
  ASSERT_EQ(static_cast<int>(dest[0]), 128569);
  ASSERT_EQ(static_cast<int>(dest[1]), 128569);
  // Making sure the pointer is not getting updated
  ASSERT_EQ(src, original);
  // Trying to read the 3rd byte should set errno
  n = LIBC_NAMESPACE::mbstowcs(dest, src + 2, 2);
  ASSERT_ERRNO_EQ(EILSEQ);
  ASSERT_EQ(static_cast<int>(n), -1);
  // Making sure the pointer is not getting updated
  ASSERT_EQ(src, original);
}

#if defined(LIBC_ADD_NULL_CHECKS)
TEST(LlvmLibcMBSToWCSTest, NullptrCrash) {
  // Passing in a nullptr should crash the program.
  EXPECT_DEATH([] { LIBC_NAMESPACE::mbstowcs(nullptr, nullptr, 1); },
               WITH_SIGNAL(-1));
}
#endif // LIBC_ADD_NULL_CHECKS
