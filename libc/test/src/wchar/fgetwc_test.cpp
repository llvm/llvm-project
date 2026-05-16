//===-- Unittests for fgetwc ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/stdint_proxy.h"
#include "hdr/wchar_macros.h" // For WEOF
#include "src/stdio/fclose.h"
#include "src/stdio/feof.h"
#include "src/stdio/ferror.h"
#include "src/stdio/fopen.h"
#include "src/stdio/fwrite.h"
#include "src/wchar/fgetwc.h"
#include "src/wchar/fwide.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcFgetwcTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcFgetwcTest, ReadASCII) {
  auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("fgetwc_ascii.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);

  // Write "abc"
  constexpr char CONTENT[] = "abc";
  ASSERT_EQ(LIBC_NAMESPACE::fwrite(CONTENT, 1, sizeof(CONTENT) - 1, file),
            sizeof(CONTENT) - 1);
  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);

  // Open for reading
  file = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);

  EXPECT_EQ(LIBC_NAMESPACE::fgetwc(file), static_cast<wint_t>(L'a'));
  EXPECT_EQ(LIBC_NAMESPACE::fgetwc(file), static_cast<wint_t>(L'b'));
  EXPECT_EQ(LIBC_NAMESPACE::fgetwc(file), static_cast<wint_t>(L'c'));

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}

TEST_F(LlvmLibcFgetwcTest, ReadUtf8) {
  auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("fgetwc_utf8.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);

  // Write "a¢€𐍈"
  // a   -> 0x61 (1-byte)
  // ¢   -> 0xC2 0xA2 (2-byte)
  // €   -> 0xE2 0x82 0xAC (3-byte)
  // 𐍈   -> 0xF0 0x90 0x8D 0x88 (4-byte)
  constexpr unsigned char CONTENT[] = {0x61, 0xC2, 0xA2, 0xE2, 0x82,
                                       0xAC, 0xF0, 0x90, 0x8D, 0x88};
  ASSERT_EQ(LIBC_NAMESPACE::fwrite(CONTENT, 1, sizeof(CONTENT), file),
            sizeof(CONTENT));
  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);

  // Open for reading
  file = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);

  EXPECT_EQ(LIBC_NAMESPACE::fgetwc(file), static_cast<wint_t>(L'a'));
  EXPECT_EQ(LIBC_NAMESPACE::fgetwc(file), static_cast<wint_t>(L'¢'));
  // Skip characters beyond 16 bits when wint_t can't fit them.
#if WINT_MAX > 0xFFFF
  EXPECT_EQ(LIBC_NAMESPACE::fgetwc(file), static_cast<wint_t>(L'€'));
  EXPECT_EQ(LIBC_NAMESPACE::fgetwc(file), static_cast<wint_t>(L'𐍈'));
#endif

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}

TEST_F(LlvmLibcFgetwcTest, EOFAndInvalidStream) {
  auto FILENAME = libc_make_test_file_path(APPEND_LIBC_TEST("fgetwc_eof.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);

  // Write nothing, just close
  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);

  // Open for reading
  file = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);

  // EOF Test
  EXPECT_EQ(LIBC_NAMESPACE::fgetwc(file), static_cast<wint_t>(WEOF));
  EXPECT_NE(LIBC_NAMESPACE::feof(file), 0);
  EXPECT_EQ(LIBC_NAMESPACE::ferror(file), 0);

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);

  // Open in write-only mode and try to read
  file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);

  ASSERT_ERRNO_SUCCESS();
  EXPECT_EQ(LIBC_NAMESPACE::fgetwc(file), static_cast<wint_t>(WEOF));
  ASSERT_ERRNO_EQ(EBADF);

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}

TEST_F(LlvmLibcFgetwcTest, EncodingErrorEILSEQ) {
  auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("fgetwc_eilseq.test"));
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
  ASSERT_ERRNO_SUCCESS();
  EXPECT_EQ(LIBC_NAMESPACE::fgetwc(file), static_cast<wint_t>(WEOF));
  ASSERT_ERRNO_EQ(EILSEQ);
  EXPECT_NE(LIBC_NAMESPACE::ferror(file), 0);

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}

TEST_F(LlvmLibcFgetwcTest, ByteModeFailure) {
  auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("fgetwc_bytemode.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w+");
  ASSERT_FALSE(file == nullptr);

  // Orient to byte mode
  EXPECT_LT(LIBC_NAMESPACE::fwide(file, -1), 0);

  // Read wide char should fail and set errno to EINVAL
  EXPECT_EQ(LIBC_NAMESPACE::fgetwc(file), static_cast<wint_t>(WEOF));
  ASSERT_ERRNO_EQ(EINVAL);
  EXPECT_NE(LIBC_NAMESPACE::ferror(file), 0);

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}
