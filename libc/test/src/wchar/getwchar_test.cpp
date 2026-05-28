//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for getwchar
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/stdint_proxy.h"
#include "hdr/wchar_macros.h" // For WEOF
#include "src/stdio/fclose.h"
#include "src/stdio/feof.h"
#include "src/stdio/ferror.h"
#include "src/stdio/fopen.h"
#include "src/stdio/fwrite.h"
#include "src/stdio/stdin.h"
#include "src/wchar/fwide.h"
#include "src/wchar/getwchar.h"
#include "src/wchar/ungetwc.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcGetwcharTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcGetwcharTest, ReadValidWideCharacters) {
  const auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("getwchar_valid.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);

  // Write "12"
  constexpr char CONTENT[] = "12";
  ASSERT_EQ(LIBC_NAMESPACE::fwrite(CONTENT, 1, sizeof(CONTENT) - 1, file),
            sizeof(CONTENT) - 1);
  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);

  // Redirect stdin
  ::FILE *original_stdin = LIBC_NAMESPACE::stdin;
  LIBC_NAMESPACE::stdin = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(LIBC_NAMESPACE::stdin == nullptr);

  // Verify unoriented
  EXPECT_EQ(LIBC_NAMESPACE::fwide(LIBC_NAMESPACE::stdin, 0), 0);

  // Read characters
  EXPECT_EQ(LIBC_NAMESPACE::getwchar(), static_cast<wint_t>(L'1'));
  EXPECT_EQ(LIBC_NAMESPACE::getwchar(), static_cast<wint_t>(L'2'));

  // Orientation should be wide
  EXPECT_GT(LIBC_NAMESPACE::fwide(LIBC_NAMESPACE::stdin, 0), 0);

  // Restore stdin
  ASSERT_EQ(LIBC_NAMESPACE::fclose(LIBC_NAMESPACE::stdin), 0);
  LIBC_NAMESPACE::stdin = original_stdin;
}

TEST_F(LlvmLibcGetwcharTest, ReadUtf8) {
  const auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("getwchar_utf8.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);

  // Write "a¢€𐍈"
  constexpr unsigned char CONTENT[] = {0x61, 0xC2, 0xA2, 0xE2, 0x82,
                                       0xAC, 0xF0, 0x90, 0x8D, 0x88};
  ASSERT_EQ(LIBC_NAMESPACE::fwrite(CONTENT, 1, sizeof(CONTENT), file),
            sizeof(CONTENT));
  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);

  // Redirect stdin
  ::FILE *original_stdin = LIBC_NAMESPACE::stdin;
  LIBC_NAMESPACE::stdin = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(LIBC_NAMESPACE::stdin == nullptr);

  EXPECT_EQ(LIBC_NAMESPACE::getwchar(), static_cast<wint_t>(L'a'));
  EXPECT_EQ(LIBC_NAMESPACE::getwchar(), static_cast<wint_t>(L'¢'));
#if WINT_MAX > 0xFFFF
  EXPECT_EQ(LIBC_NAMESPACE::getwchar(), static_cast<wint_t>(L'€'));
  EXPECT_EQ(LIBC_NAMESPACE::getwchar(), static_cast<wint_t>(L'𐍈'));
#endif

  // Restore stdin
  ASSERT_EQ(LIBC_NAMESPACE::fclose(LIBC_NAMESPACE::stdin), 0);
  LIBC_NAMESPACE::stdin = original_stdin;
}

TEST_F(LlvmLibcGetwcharTest, EndOfFileBehavior) {
  const auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("getwchar_eof.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);
  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);

  // Redirect stdin
  ::FILE *original_stdin = LIBC_NAMESPACE::stdin;
  LIBC_NAMESPACE::stdin = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(LIBC_NAMESPACE::stdin == nullptr);

  // Read past EOF
  EXPECT_EQ(LIBC_NAMESPACE::getwchar(), static_cast<wint_t>(WEOF));
  EXPECT_NE(LIBC_NAMESPACE::feof(LIBC_NAMESPACE::stdin), 0);
  EXPECT_EQ(LIBC_NAMESPACE::ferror(LIBC_NAMESPACE::stdin), 0);

  // Restore stdin
  ASSERT_EQ(LIBC_NAMESPACE::fclose(LIBC_NAMESPACE::stdin), 0);
  LIBC_NAMESPACE::stdin = original_stdin;
}

TEST_F(LlvmLibcGetwcharTest, ReadError) {
  const auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("getwchar_readerr.test"));

  // Redirect stdin using a write-only stream
  ::FILE *original_stdin = LIBC_NAMESPACE::stdin;
  LIBC_NAMESPACE::stdin = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(LIBC_NAMESPACE::stdin == nullptr);

  // Try to read from write-only stream
  ASSERT_ERRNO_SUCCESS();
  EXPECT_EQ(LIBC_NAMESPACE::getwchar(), static_cast<wint_t>(WEOF));
  ASSERT_ERRNO_EQ(EBADF);
  EXPECT_NE(LIBC_NAMESPACE::ferror(LIBC_NAMESPACE::stdin), 0);

  // Restore stdin
  ASSERT_EQ(LIBC_NAMESPACE::fclose(LIBC_NAMESPACE::stdin), 0);
  LIBC_NAMESPACE::stdin = original_stdin;
}

TEST_F(LlvmLibcGetwcharTest, ByteOrientedMisuse) {
  const auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("getwchar_bytemode.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w+");
  ASSERT_FALSE(file == nullptr);

  // Redirect stdin
  ::FILE *original_stdin = LIBC_NAMESPACE::stdin;
  LIBC_NAMESPACE::stdin = file;

  // Orient to byte mode
  EXPECT_LT(LIBC_NAMESPACE::fwide(LIBC_NAMESPACE::stdin, -1), 0);

  // Reading should fail and set errno to EINVAL
  EXPECT_EQ(LIBC_NAMESPACE::getwchar(), static_cast<wint_t>(WEOF));
  ASSERT_ERRNO_EQ(EINVAL);
  EXPECT_NE(LIBC_NAMESPACE::ferror(LIBC_NAMESPACE::stdin), 0);

  // Restore stdin
  ASSERT_EQ(LIBC_NAMESPACE::fclose(LIBC_NAMESPACE::stdin), 0);
  LIBC_NAMESPACE::stdin = original_stdin;
}

TEST_F(LlvmLibcGetwcharTest, InteractionWithUngetwc) {
  const auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("getwchar_unget.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);

  // Write "1"
  constexpr char CONTENT[] = "1";
  ASSERT_EQ(LIBC_NAMESPACE::fwrite(CONTENT, 1, sizeof(CONTENT) - 1, file),
            sizeof(CONTENT) - 1);
  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);

  // Redirect stdin
  ::FILE *original_stdin = LIBC_NAMESPACE::stdin;
  LIBC_NAMESPACE::stdin = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(LIBC_NAMESPACE::stdin == nullptr);

  // Read '1'
  EXPECT_EQ(LIBC_NAMESPACE::getwchar(), static_cast<wint_t>(L'1'));

  // Push back 'X'
  EXPECT_EQ(LIBC_NAMESPACE::ungetwc(L'X', LIBC_NAMESPACE::stdin),
            static_cast<wint_t>(L'X'));

  // Read again -> should be 'X'
  EXPECT_EQ(LIBC_NAMESPACE::getwchar(), static_cast<wint_t>(L'X'));

  // Restore stdin
  ASSERT_EQ(LIBC_NAMESPACE::fclose(LIBC_NAMESPACE::stdin), 0);
  LIBC_NAMESPACE::stdin = original_stdin;
}
