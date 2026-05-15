//===-- Unittests for getwc -----------------------------------------------===//
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
#include "src/wchar/fwide.h"
#include "src/wchar/getwc.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcGetwcTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcGetwcTest, ReadValidCharacters) {
  auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("getwc_valid.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);

  // Write "12"
  constexpr char CONTENT[] = "12";
  ASSERT_EQ(LIBC_NAMESPACE::fwrite(CONTENT, 1, sizeof(CONTENT) - 1, file),
            sizeof(CONTENT) - 1);
  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);

  // Open for reading
  file = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);

  // Initial orientation
  EXPECT_EQ(LIBC_NAMESPACE::fwide(file, 0), 0);

  EXPECT_EQ(LIBC_NAMESPACE::getwc(file), static_cast<wint_t>(L'1'));
  EXPECT_EQ(LIBC_NAMESPACE::getwc(file), static_cast<wint_t>(L'2'));

  // Stream orientation should now be wide
  EXPECT_GT(LIBC_NAMESPACE::fwide(file, 0), 0);

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}

TEST_F(LlvmLibcGetwcTest, ReadUtf8) {
  auto FILENAME = libc_make_test_file_path(APPEND_LIBC_TEST("getwc_utf8.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);

  // Write "a¢€𐍈"
  constexpr unsigned char CONTENT[] = {0x61, 0xC2, 0xA2, 0xE2, 0x82,
                                       0xAC, 0xF0, 0x90, 0x8D, 0x88};
  ASSERT_EQ(LIBC_NAMESPACE::fwrite(CONTENT, 1, sizeof(CONTENT), file),
            sizeof(CONTENT));
  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);

  // Open for reading
  file = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);

  EXPECT_EQ(LIBC_NAMESPACE::getwc(file), static_cast<wint_t>(L'a'));
  EXPECT_EQ(LIBC_NAMESPACE::getwc(file), static_cast<wint_t>(L'¢'));
#if WINT_MAX > 0xFFFF
  EXPECT_EQ(LIBC_NAMESPACE::getwc(file), static_cast<wint_t>(L'€'));
  EXPECT_EQ(LIBC_NAMESPACE::getwc(file), static_cast<wint_t>(L'𐍈'));
#endif

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}

TEST_F(LlvmLibcGetwcTest, EndOfFile) {
  auto FILENAME = libc_make_test_file_path(APPEND_LIBC_TEST("getwc_eof.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);
  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);

  file = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);

  // Read past EOF
  EXPECT_EQ(LIBC_NAMESPACE::getwc(file), static_cast<wint_t>(WEOF));
  EXPECT_NE(LIBC_NAMESPACE::feof(file), 0);
  EXPECT_EQ(LIBC_NAMESPACE::ferror(file), 0);

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}

TEST_F(LlvmLibcGetwcTest, ReadError) {
  auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("getwc_readerr.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);

  // Try to read from write-only file
  ASSERT_ERRNO_SUCCESS();
  EXPECT_EQ(LIBC_NAMESPACE::getwc(file), static_cast<wint_t>(WEOF));
  ASSERT_ERRNO_EQ(EBADF);
  EXPECT_NE(LIBC_NAMESPACE::ferror(file), 0);

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}

TEST_F(LlvmLibcGetwcTest, ByteOrientedStreamFail) {
  auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("getwc_bytemode.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w+");
  ASSERT_FALSE(file == nullptr);

  // Orient to byte mode
  EXPECT_LT(LIBC_NAMESPACE::fwide(file, -1), 0);

  // Reading should fail and set errno to EINVAL
  EXPECT_EQ(LIBC_NAMESPACE::getwc(file), static_cast<wint_t>(WEOF));
  ASSERT_ERRNO_EQ(EINVAL);
  EXPECT_NE(LIBC_NAMESPACE::ferror(file), 0);

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}
