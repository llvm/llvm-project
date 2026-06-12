//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for fputws
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/stdio/fclose.h"
#include "src/stdio/ferror.h"
#include "src/stdio/fopen.h"
#include "src/stdio/fread.h"
#include "src/wchar/fputws.h"
#include "src/wchar/fwide.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcFputwsTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcFputwsTest, WriteWideString) {
  const auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("fputws_string.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);

  constexpr wchar_t STR[] = L"Hello, ¢ world!\n";
#if WCHAR_MAX > 0xFFFF
  // if they're available, try 3 and 4 byte wchars.
  constexpr wchar_t STR2[] = L"€𐍈";
  constexpr size_t EXPECTED_BYTES = 24;
#else
  constexpr wchar_t STR2[] = L"";
  constexpr size_t EXPECTED_BYTES = 17;
#endif
  EXPECT_GE(LIBC_NAMESPACE::fputws(STR, file), 0);
  EXPECT_GE(LIBC_NAMESPACE::fputws(STR2, file), 0);

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);

  // Open to read raw bytes and verify UTF-8 mapping
  file = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);

  unsigned char buffer[50] = {0};
  ASSERT_EQ(LIBC_NAMESPACE::fread(buffer, 1, EXPECTED_BYTES, file),
            EXPECTED_BYTES);

  // "Hello, "
  EXPECT_EQ(buffer[0], static_cast<unsigned char>('H'));
  EXPECT_EQ(buffer[1], static_cast<unsigned char>('e'));
  EXPECT_EQ(buffer[2], static_cast<unsigned char>('l'));
  EXPECT_EQ(buffer[3], static_cast<unsigned char>('l'));
  EXPECT_EQ(buffer[4], static_cast<unsigned char>('o'));
  EXPECT_EQ(buffer[5], static_cast<unsigned char>(','));
  EXPECT_EQ(buffer[6], static_cast<unsigned char>(' '));

  // "¢"
  EXPECT_EQ(buffer[7], static_cast<unsigned char>(0xC2));
  EXPECT_EQ(buffer[8], static_cast<unsigned char>(0xA2));

  // " world!\n"
  EXPECT_EQ(buffer[9], static_cast<unsigned char>(' '));
  EXPECT_EQ(buffer[10], static_cast<unsigned char>('w'));
  EXPECT_EQ(buffer[11], static_cast<unsigned char>('o'));
  EXPECT_EQ(buffer[12], static_cast<unsigned char>('r'));
  EXPECT_EQ(buffer[13], static_cast<unsigned char>('l'));
  EXPECT_EQ(buffer[14], static_cast<unsigned char>('d'));
  EXPECT_EQ(buffer[15], static_cast<unsigned char>('!'));
  EXPECT_EQ(buffer[16], static_cast<unsigned char>('\n'));

#if WCHAR_MAX > 0xFFFF
  // "€"
  EXPECT_EQ(buffer[17], static_cast<unsigned char>(0xE2));
  EXPECT_EQ(buffer[18], static_cast<unsigned char>(0x82));
  EXPECT_EQ(buffer[19], static_cast<unsigned char>(0xAC));

  // "𐍈"
  EXPECT_EQ(buffer[20], static_cast<unsigned char>(0xF0));
  EXPECT_EQ(buffer[21], static_cast<unsigned char>(0x90));
  EXPECT_EQ(buffer[22], static_cast<unsigned char>(0x8D));
  EXPECT_EQ(buffer[23], static_cast<unsigned char>(0x88));
#endif

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}

TEST_F(LlvmLibcFputwsTest, EmptyString) {
  const auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("fputws_empty.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);

  EXPECT_GE(LIBC_NAMESPACE::fputws(L"", file), 0);

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);

  // Verify nothing was written
  file = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);

  unsigned char buffer[5] = {0};
  ASSERT_EQ(LIBC_NAMESPACE::fread(buffer, 1, 5, file), size_t(0));

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}

TEST_F(LlvmLibcFputwsTest, InvalidStream) {
  const auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("fputws_invalid.test"));

  // Create the file first
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);
  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);

  // Open in read-only mode
  file = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);

  // Try to write to read-only file
  ASSERT_ERRNO_SUCCESS();
  EXPECT_LT(LIBC_NAMESPACE::fputws(L"fail", file), 0);
  ASSERT_ERRNO_EQ(EBADF);

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}

#if WCHAR_MAX > 0xFFFF
TEST_F(LlvmLibcFputwsTest, EncodingErrorEILSEQ) {
  const auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("fputws_eilseq.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);

  // String with invalid wide character point (outside Unicode range) which
  // fails encoding
  constexpr wchar_t STR[] = {static_cast<wchar_t>(0x110000), L'\0'};
  ASSERT_ERRNO_SUCCESS();
  EXPECT_LT(LIBC_NAMESPACE::fputws(STR, file), 0);
  ASSERT_ERRNO_EQ(EILSEQ);
  EXPECT_NE(LIBC_NAMESPACE::ferror(file), 0);

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}
#endif

TEST_F(LlvmLibcFputwsTest, ByteModeFailure) {
  const auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("fputws_bytemode.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w+");
  ASSERT_FALSE(file == nullptr);

  // Orient to byte mode
  EXPECT_LT(LIBC_NAMESPACE::fwide(file, -1), 0);

  // Writing wide string should fail and set errno to EINVAL
  EXPECT_LT(LIBC_NAMESPACE::fputws(L"fail", file), 0);
  ASSERT_ERRNO_EQ(EINVAL);

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}
