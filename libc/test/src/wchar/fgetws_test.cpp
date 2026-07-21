//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for fgetws
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/stdint_proxy.h"
#include "hdr/types/wint_t.h"
#include "src/stdio/fclose.h"
#include "src/stdio/ferror.h"
#include "src/stdio/fopen.h"
#include "src/stdio/fwrite.h"
#include "src/wchar/fgetws.h"
#include "src/wchar/fwide.h"
#include "src/wchar/wcscmp.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcFgetwsTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

// TODO: Refactor these tests to use standard wide-character string comparison
// assert macros (e.g., EXPECT_STREQ for wchar_t) once they are supported
// natively by the LLVM-libc test framework, instead of calling wcscmp.
TEST_F(LlvmLibcFgetwsTest, ReadWideString) {
  const auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("fgetws_string.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);

#if WCHAR_MAX > 0xFFFF
  // Write UTF-8 bytes for: "Hello, ¢€𐍈 world!\n"
  constexpr unsigned char CONTENT[] = {
      'H',  'e',  'l',  'l',  'o', ',', ' ', 0xC2, 0xA2, 0xE2, 0x82, 0xAC,
      0xF0, 0x90, 0x8D, 0x88, ' ', 'w', 'o', 'r',  'l',  'd',  '!',  '\n'};
  constexpr const wchar_t *EXPECTED_STR = L"Hello, ¢€𐍈 world!\n";
#else
  // Write UTF-8 bytes for: "Hello, ¢ world!\n"
  constexpr unsigned char CONTENT[] = {'H', 'e',  'l',  'l', 'o', ',',
                                       ' ', 0xC2, 0xA2, ' ', 'w', 'o',
                                       'r', 'l',  'd',  '!', '\n'};
  constexpr const wchar_t *EXPECTED_STR = L"Hello, ¢ world!\n";
#endif
  ASSERT_EQ(LIBC_NAMESPACE::fwrite(CONTENT, 1, sizeof(CONTENT), file),
            sizeof(CONTENT));
  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);

  // Open for reading
  file = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);

  wchar_t buffer[50] = {0};
  wchar_t *result = LIBC_NAMESPACE::fgetws(buffer, 50, file);
  ASSERT_FALSE(result == nullptr);
  EXPECT_EQ(result, buffer);
  EXPECT_EQ(LIBC_NAMESPACE::wcscmp(buffer, EXPECTED_STR), 0);

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}

TEST_F(LlvmLibcFgetwsTest, ReadBounded) {
  const auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("fgetws_bounded.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);

  // Write "1234567890"
  constexpr char CONTENT[] = "1234567890";
  ASSERT_EQ(LIBC_NAMESPACE::fwrite(CONTENT, 1, sizeof(CONTENT) - 1, file),
            sizeof(CONTENT) - 1);
  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);

  // Open for reading
  file = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);

  wchar_t buffer[10] = {0};
  // Read bounded by count = 5 (4 chars + null terminator)
  wchar_t *result = LIBC_NAMESPACE::fgetws(buffer, 5, file);
  ASSERT_FALSE(result == nullptr);
  EXPECT_EQ(result, buffer);
  EXPECT_EQ(LIBC_NAMESPACE::wcscmp(buffer, L"1234"), 0);

  // Read bounded by count = 1 (writes only null terminator)
  wchar_t buffer_one[5] = {L'x', L'y', L'z', L'\0'};
  wchar_t *result_one = LIBC_NAMESPACE::fgetws(buffer_one, 1, file);
  ASSERT_FALSE(result_one == nullptr);
  EXPECT_EQ(result_one, buffer_one);
  EXPECT_EQ(static_cast<wint_t>(buffer_one[0]), static_cast<wint_t>(L'\0'));
  EXPECT_EQ(static_cast<wint_t>(buffer_one[1]),
            static_cast<wint_t>(L'y')); // untouched

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}

TEST_F(LlvmLibcFgetwsTest, NewlineStops) {
  const auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("fgetws_newline.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);

  // Write "abc\ndef"
  constexpr char CONTENT[] = "abc\ndef";
  ASSERT_EQ(LIBC_NAMESPACE::fwrite(CONTENT, 1, sizeof(CONTENT) - 1, file),
            sizeof(CONTENT) - 1);
  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);

  // Open for reading
  file = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);

  wchar_t buffer[20] = {0};
  wchar_t *result = LIBC_NAMESPACE::fgetws(buffer, 10, file);
  ASSERT_FALSE(result == nullptr);
  EXPECT_EQ(result, buffer);
  EXPECT_EQ(LIBC_NAMESPACE::wcscmp(buffer, L"abc\n"), 0);

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}

TEST_F(LlvmLibcFgetwsTest, InvalidStream) {
  const auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("fgetws_invalid.test"));

  // Create the file first
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);
  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);

  // Open in write-only mode
  file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);

  // Try to read from write-only stream
  wchar_t buffer[10] = {0};
  ASSERT_ERRNO_SUCCESS();
  EXPECT_EQ(LIBC_NAMESPACE::fgetws(buffer, 5, file),
            static_cast<wchar_t *>(nullptr));
  ASSERT_ERRNO_EQ(EBADF);
  EXPECT_NE(LIBC_NAMESPACE::ferror(file), 0);

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}

TEST_F(LlvmLibcFgetwsTest, EncodingErrorEILSEQ) {
  const auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("fgetws_eilseq.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);

  // Write an invalid UTF-8 sequence: 0x80 (stray continuation byte)
  constexpr unsigned char CONTENT[] = {0x80};
  ASSERT_EQ(LIBC_NAMESPACE::fwrite(CONTENT, 1, sizeof(CONTENT), file),
            sizeof(CONTENT));
  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);

  // Open for reading
  file = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);

  // Reading invalid sequence should fail with EILSEQ
  wchar_t buffer[10] = {0};
  ASSERT_ERRNO_SUCCESS();
  EXPECT_EQ(LIBC_NAMESPACE::fgetws(buffer, 5, file),
            static_cast<wchar_t *>(nullptr));
  ASSERT_ERRNO_EQ(EILSEQ);
  EXPECT_NE(LIBC_NAMESPACE::ferror(file), 0);

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}

TEST_F(LlvmLibcFgetwsTest, ByteModeFailure) {
  const auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("fgetws_bytemode.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w+");
  ASSERT_FALSE(file == nullptr);

  // Orient to byte mode
  EXPECT_LT(LIBC_NAMESPACE::fwide(file, -1), 0);

  // Read wide string should fail and set errno to EINVAL
  wchar_t buffer[10] = {0};
  EXPECT_EQ(LIBC_NAMESPACE::fgetws(buffer, 5, file),
            static_cast<wchar_t *>(nullptr));
  ASSERT_ERRNO_EQ(EINVAL);
  EXPECT_NE(LIBC_NAMESPACE::ferror(file), 0);

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}
