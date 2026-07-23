//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for fputwc
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/stdint_proxy.h"
#include "hdr/wchar_macros.h" // For WEOF
#include "src/stdio/fclose.h"
#include "src/stdio/ferror.h"
#include "src/stdio/fopen.h"
#include "src/stdio/fread.h"
#include "src/stdio/fwrite.h"
#include "src/wchar/fputwc.h"
#include "src/wchar/fwide.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcFputwcTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcFputwcTest, WriteASCII) {
  const auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("fputwc_ascii.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);

  // Write 'a', 'b', 'c'
  EXPECT_EQ(LIBC_NAMESPACE::fputwc(L'a', file), static_cast<wint_t>(L'a'));
  EXPECT_EQ(LIBC_NAMESPACE::fputwc(L'b', file), static_cast<wint_t>(L'b'));
  EXPECT_EQ(LIBC_NAMESPACE::fputwc(L'c', file), static_cast<wint_t>(L'c'));

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);

  // Open again to read raw bytes
  file = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);

  char buffer[10] = {0};
  ASSERT_EQ(LIBC_NAMESPACE::fread(buffer, 1, 3, file), size_t(3));
  EXPECT_STREQ(buffer, "abc");

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}

TEST_F(LlvmLibcFputwcTest, WriteUtf8) {
  const auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("fputwc_utf8.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);

  // 1-byte character: 'a' (L'a', 0x61) -> UTF-8: 0x61
  EXPECT_EQ(LIBC_NAMESPACE::fputwc(L'a', file), static_cast<wint_t>(L'a'));

  // 2-byte character: '¢' (L'¢', 0xA2) -> UTF-8: 0xC2 0xA2
  EXPECT_EQ(LIBC_NAMESPACE::fputwc(L'¢', file), static_cast<wint_t>(L'¢'));

#if WCHAR_MAX > 0xFFFF
  // 3-byte character: '€' (L'€', 0x20AC) -> UTF-8: 0xE2 0x82 0xAC
  EXPECT_EQ(LIBC_NAMESPACE::fputwc(L'€', file), static_cast<wint_t>(L'€'));

  // 4-byte character: '𐍈' (L'𐍈', 0x10348) -> UTF-8: 0xF0 0x90 0x8D 0x88
  EXPECT_EQ(LIBC_NAMESPACE::fputwc(L'𐍈', file), static_cast<wint_t>(L'𐍈'));
#endif
  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);

  // Open again to read raw bytes
  file = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);

  unsigned char buffer[15] = {0};
  ASSERT_EQ(LIBC_NAMESPACE::fread(buffer, 1, 10, file), size_t(10));

  // Verify 1-byte
  EXPECT_EQ(buffer[0], static_cast<unsigned char>(0x61));

  // Verify 2-byte
  EXPECT_EQ(buffer[1], static_cast<unsigned char>(0xC2));
  EXPECT_EQ(buffer[2], static_cast<unsigned char>(0xA2));

#if WCHAR_MAX > 0xFFFF
  // Verify 3-byte
  EXPECT_EQ(buffer[3], static_cast<unsigned char>(0xE2));
  EXPECT_EQ(buffer[4], static_cast<unsigned char>(0x82));
  EXPECT_EQ(buffer[5], static_cast<unsigned char>(0xAC));

  // Verify 4-byte
  EXPECT_EQ(buffer[6], static_cast<unsigned char>(0xF0));
  EXPECT_EQ(buffer[7], static_cast<unsigned char>(0x90));
  EXPECT_EQ(buffer[8], static_cast<unsigned char>(0x8D));
  EXPECT_EQ(buffer[9], static_cast<unsigned char>(0x88));
#endif
  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}

// For the character to be outside the unicode range it also needs to be outside
// the UTF-16 range.
#if WCHAR_MAX > 0xFFFF
TEST_F(LlvmLibcFputwcTest, EncodingErrorEILSEQ) {
  const auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("fputwc_eilseq.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);

  // Try to write an invalid wide character point (outside Unicode range)
  ASSERT_ERRNO_SUCCESS();
  EXPECT_EQ(LIBC_NAMESPACE::fputwc(static_cast<wchar_t>(0x110000), file),
            static_cast<wint_t>(WEOF));
  ASSERT_ERRNO_EQ(EILSEQ);
  EXPECT_NE(LIBC_NAMESPACE::ferror(file), 0);

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}
#endif

TEST_F(LlvmLibcFputwcTest, InvalidStream) {
  const auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("fputwc_invalid.test"));

  // Create the file first
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);
  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);

  // Open in read-only mode
  file = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);

  // Try to write to read-only file
  ASSERT_ERRNO_SUCCESS();
  EXPECT_EQ(LIBC_NAMESPACE::fputwc(L'x', file), static_cast<wint_t>(WEOF));
  ASSERT_ERRNO_EQ(EBADF);

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}

TEST_F(LlvmLibcFputwcTest, ByteModeFailure) {
  const auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("fputwc_bytemode.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);

  // Orient to byte mode
  EXPECT_LT(LIBC_NAMESPACE::fwide(file, -1), 0);

  // Writing wide char should fail and set errno to EINVAL
  EXPECT_EQ(LIBC_NAMESPACE::fputwc(L'a', file), static_cast<wint_t>(WEOF));
  ASSERT_ERRNO_EQ(EINVAL);

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}
