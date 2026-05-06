//===-- Unittests for ungetwc ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/stdio_macros.h" // For SEEK_CUR
#include "hdr/wchar_macros.h" // For WEOF
#include "src/stdio/fclose.h"
#include "src/stdio/feof.h"
#include "src/stdio/ferror.h"
#include "src/stdio/fopen.h"
#include "src/stdio/fseek.h"
#include "src/stdio/fwrite.h"
#include "src/wchar/fgetwc.h"
#include "src/wchar/fwide.h"
#include "src/wchar/ungetwc.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcUngetwcTest, PushBackAndRead) {
  auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("ungetwc_push.test"));
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

  // Read first char -> '1'
  EXPECT_EQ(LIBC_NAMESPACE::fgetwc(file), static_cast<wint_t>(L'1'));

  // Push back 'X'
  EXPECT_EQ(LIBC_NAMESPACE::ungetwc(L'X', file), static_cast<wint_t>(L'X'));

  // Read again -> should get 'X'
  EXPECT_EQ(LIBC_NAMESPACE::fgetwc(file), static_cast<wint_t>(L'X'));

  // Read again -> should get '2'
  EXPECT_EQ(LIBC_NAMESPACE::fgetwc(file), static_cast<wint_t>(L'2'));

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}

TEST(LlvmLibcUngetwcTest, PushBackWEOF) {
  auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("ungetwc_weof.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);
  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);

  file = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);

  // Push back WEOF should do nothing and return WEOF
  EXPECT_EQ(LIBC_NAMESPACE::ungetwc(WEOF, file), static_cast<wint_t>(WEOF));

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}

TEST(LlvmLibcUngetwcTest, ByteModeFailure) {
  auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("ungetwc_bytemode.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w+");
  ASSERT_FALSE(file == nullptr);

  // Orient to byte mode
  EXPECT_LT(LIBC_NAMESPACE::fwide(file, -1), 0);

  // Push back should fail
  EXPECT_EQ(LIBC_NAMESPACE::ungetwc(L'a', file), static_cast<wint_t>(WEOF));

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}

TEST(LlvmLibcUngetwcTest, OrientUnorientedStream) {
  auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("ungetwc_orient.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w+");
  ASSERT_FALSE(file == nullptr);

  // Verify initial unoriented state
  EXPECT_EQ(LIBC_NAMESPACE::fwide(file, 0), 0);

  // Push back a char
  EXPECT_EQ(LIBC_NAMESPACE::ungetwc(L'A', file), static_cast<wint_t>(L'A'));

  // Verify stream is now wide-oriented
  EXPECT_GT(LIBC_NAMESPACE::fwide(file, 0), 0);

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}

TEST(LlvmLibcUngetwcTest, ClearEofIndicator) {
  auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("ungetwc_cleareof.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);

  // Write "1"
  constexpr char CONTENT[] = "1";
  ASSERT_EQ(LIBC_NAMESPACE::fwrite(CONTENT, 1, sizeof(CONTENT) - 1, file),
            sizeof(CONTENT) - 1);
  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);

  file = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);

  // Read '1'
  EXPECT_EQ(LIBC_NAMESPACE::fgetwc(file), static_cast<wint_t>(L'1'));

  // Read past EOF
  EXPECT_EQ(LIBC_NAMESPACE::fgetwc(file), static_cast<wint_t>(WEOF));
  EXPECT_NE(LIBC_NAMESPACE::feof(file), 0);

  // ungetwc should clear EOF indicator
  EXPECT_EQ(LIBC_NAMESPACE::ungetwc(L'1', file), static_cast<wint_t>(L'1'));
  EXPECT_EQ(LIBC_NAMESPACE::feof(file), 0);
  EXPECT_EQ(LIBC_NAMESPACE::ferror(file), 0); // error remains unmodified

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}

TEST(LlvmLibcUngetwcTest, DiscardOnFilePositioning) {
  auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("ungetwc_discard.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);

  // Write "12"
  constexpr char CONTENT[] = "12";
  ASSERT_EQ(LIBC_NAMESPACE::fwrite(CONTENT, 1, sizeof(CONTENT) - 1, file),
            sizeof(CONTENT) - 1);
  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);

  file = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);

  // Read first char -> '1'
  EXPECT_EQ(LIBC_NAMESPACE::fgetwc(file), static_cast<wint_t>(L'1'));

  // Push back 'X'
  EXPECT_EQ(LIBC_NAMESPACE::ungetwc(L'X', file), static_cast<wint_t>(L'X'));

  // Seek to absolute position 1 (after '1') to discard pushed-back char
  EXPECT_EQ(LIBC_NAMESPACE::fseek(file, 1, SEEK_SET), 0);

  // Read again -> should get '2' instead of 'X'
  EXPECT_EQ(LIBC_NAMESPACE::fgetwc(file), static_cast<wint_t>(L'2'));

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}

TEST(LlvmLibcUngetwcTest, LifoMultiplePushbacks) {
  auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("ungetwc_lifo.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w+");
  ASSERT_FALSE(file == nullptr);

  // Push back 'X' then 'Y'
  EXPECT_EQ(LIBC_NAMESPACE::ungetwc(L'X', file), static_cast<wint_t>(L'X'));
  // If multiple push-backs are supported, verify LIFO ordering
  wint_t second_push = LIBC_NAMESPACE::ungetwc(L'Y', file);
  if (second_push == static_cast<wint_t>(L'Y')) {
    EXPECT_EQ(LIBC_NAMESPACE::fgetwc(file), static_cast<wint_t>(L'Y'));
    EXPECT_EQ(LIBC_NAMESPACE::fgetwc(file), static_cast<wint_t>(L'X'));
  } else {
    // Only one push-back supported on this platform/implementation
    EXPECT_EQ(LIBC_NAMESPACE::fgetwc(file), static_cast<wint_t>(L'X'));
  }

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}

TEST(LlvmLibcUngetwcTest, PushbackAtStartOfFile) {
  auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("ungetwc_start.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);

  // Write "1"
  constexpr char CONTENT[] = "1";
  ASSERT_EQ(LIBC_NAMESPACE::fwrite(CONTENT, 1, sizeof(CONTENT) - 1, file),
            sizeof(CONTENT) - 1);
  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);

  file = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);

  // Push back 'Z' at the very beginning without prior reads
  EXPECT_EQ(LIBC_NAMESPACE::ungetwc(L'Z', file), static_cast<wint_t>(L'Z'));

  // Read first char -> should be 'Z'
  EXPECT_EQ(LIBC_NAMESPACE::fgetwc(file), static_cast<wint_t>(L'Z'));

  // Read next char -> should be '1'
  EXPECT_EQ(LIBC_NAMESPACE::fgetwc(file), static_cast<wint_t>(L'1'));

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}
