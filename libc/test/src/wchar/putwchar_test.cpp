//===-- Unittests for putwchar --------------------------------------------===//
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
#include "src/stdio/ferror.h"
#include "src/stdio/fopen.h"
#include "src/stdio/fread.h"
#include "src/stdio/stdout.h"
#include "src/wchar/fwide.h"
#include "src/wchar/putwchar.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcPutwcharTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcPutwcharTest, WriteASCII) {
  auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("putwchar_ascii.test"));

  // Redirect stdout
  ::FILE *original_stdout = LIBC_NAMESPACE::stdout;
  LIBC_NAMESPACE::stdout = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(LIBC_NAMESPACE::stdout == nullptr);

  // Initial orientation
  EXPECT_EQ(LIBC_NAMESPACE::fwide(LIBC_NAMESPACE::stdout, 0), 0);

  EXPECT_EQ(LIBC_NAMESPACE::putwchar(L'a'), static_cast<wint_t>(L'a'));
  EXPECT_EQ(LIBC_NAMESPACE::putwchar(L'b'), static_cast<wint_t>(L'b'));
  EXPECT_EQ(LIBC_NAMESPACE::putwchar(L'c'), static_cast<wint_t>(L'c'));

  // Orientation should be wide
  EXPECT_GT(LIBC_NAMESPACE::fwide(LIBC_NAMESPACE::stdout, 0), 0);

  // Restore stdout
  ASSERT_EQ(LIBC_NAMESPACE::fclose(LIBC_NAMESPACE::stdout), 0);
  LIBC_NAMESPACE::stdout = original_stdout;

  // Read back raw bytes to verify
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);

  unsigned char buffer[5] = {0};
  ASSERT_EQ(LIBC_NAMESPACE::fread(buffer, 1, 3, file), size_t(3));
  EXPECT_EQ(buffer[0], static_cast<unsigned char>('a'));
  EXPECT_EQ(buffer[1], static_cast<unsigned char>('b'));
  EXPECT_EQ(buffer[2], static_cast<unsigned char>('c'));

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}

TEST_F(LlvmLibcPutwcharTest, WriteUtf8) {
  auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("putwchar_utf8.test"));

  // Redirect stdout
  ::FILE *original_stdout = LIBC_NAMESPACE::stdout;
  LIBC_NAMESPACE::stdout = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(LIBC_NAMESPACE::stdout == nullptr);

  // a   -> L'a' (1-byte)
  // ¢   -> L'¢' (2-byte)
  // €   -> L'€' (3-byte)
  // 𐍈   -> L'𐍈' (4-byte)
  EXPECT_EQ(LIBC_NAMESPACE::putwchar(L'a'), static_cast<wint_t>(L'a'));
  EXPECT_EQ(LIBC_NAMESPACE::putwchar(L'¢'), static_cast<wint_t>(L'¢'));
#if WINT_MAX > 0xFFFF
  EXPECT_EQ(LIBC_NAMESPACE::putwchar(L'€'), static_cast<wint_t>(L'€'));
  EXPECT_EQ(LIBC_NAMESPACE::putwchar(L'𐍈'), static_cast<wint_t>(L'𐍈'));
  constexpr size_t EXPECTED_BYTES = 10;
#else
  constexpr size_t EXPECTED_BYTES = 3;
#endif

  // Restore stdout
  ASSERT_EQ(LIBC_NAMESPACE::fclose(LIBC_NAMESPACE::stdout), 0);
  LIBC_NAMESPACE::stdout = original_stdout;

  // Read back raw bytes and verify UTF-8 sequences exactly on disk
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);

  unsigned char buffer[20] = {0};
  ASSERT_EQ(LIBC_NAMESPACE::fread(buffer, 1, EXPECTED_BYTES, file),
            EXPECTED_BYTES);

  // a
  EXPECT_EQ(buffer[0], static_cast<unsigned char>(0x61));

  // ¢
  EXPECT_EQ(buffer[1], static_cast<unsigned char>(0xC2));
  EXPECT_EQ(buffer[2], static_cast<unsigned char>(0xA2));

#if WINT_MAX > 0xFFFF
  // €
  EXPECT_EQ(buffer[3], static_cast<unsigned char>(0xE2));
  EXPECT_EQ(buffer[4], static_cast<unsigned char>(0x82));
  EXPECT_EQ(buffer[5], static_cast<unsigned char>(0xAC));

  // 𐍈
  EXPECT_EQ(buffer[6], static_cast<unsigned char>(0xF0));
  EXPECT_EQ(buffer[7], static_cast<unsigned char>(0x90));
  EXPECT_EQ(buffer[8], static_cast<unsigned char>(0x8D));
  EXPECT_EQ(buffer[9], static_cast<unsigned char>(0x88));
#endif

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}

TEST_F(LlvmLibcPutwcharTest, InvalidStream) {
  auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("putwchar_invalid.test"));

  // Create the file first
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);
  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);

  // Redirect stdout using a read-only file stream
  ::FILE *original_stdout = LIBC_NAMESPACE::stdout;
  LIBC_NAMESPACE::stdout = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(LIBC_NAMESPACE::stdout == nullptr);

  // Try to write to read-only stdout
  ASSERT_ERRNO_SUCCESS();
  EXPECT_EQ(LIBC_NAMESPACE::putwchar(L'a'), static_cast<wint_t>(WEOF));
  ASSERT_ERRNO_EQ(EBADF);
  EXPECT_NE(LIBC_NAMESPACE::ferror(LIBC_NAMESPACE::stdout), 0);

  // Restore stdout
  ASSERT_EQ(LIBC_NAMESPACE::fclose(LIBC_NAMESPACE::stdout), 0);
  LIBC_NAMESPACE::stdout = original_stdout;
}

TEST_F(LlvmLibcPutwcharTest, ByteModeFailure) {
  auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("putwchar_bytemode.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w+");
  ASSERT_FALSE(file == nullptr);

  // Redirect stdout
  ::FILE *original_stdout = LIBC_NAMESPACE::stdout;
  LIBC_NAMESPACE::stdout = file;

  // Orient to byte mode
  EXPECT_LT(LIBC_NAMESPACE::fwide(LIBC_NAMESPACE::stdout, -1), 0);

  // Writing should fail and set errno to EINVAL
  EXPECT_EQ(LIBC_NAMESPACE::putwchar(L'a'), static_cast<wint_t>(WEOF));
  ASSERT_ERRNO_EQ(EINVAL);
  EXPECT_NE(LIBC_NAMESPACE::ferror(LIBC_NAMESPACE::stdout), 0);

  // Restore stdout
  ASSERT_EQ(LIBC_NAMESPACE::fclose(LIBC_NAMESPACE::stdout), 0);
  LIBC_NAMESPACE::stdout = original_stdout;
}

TEST_F(LlvmLibcPutwcharTest, RealStdoutNoRedirection) {
  // This test writes directly to the real un-redirected stdout.
  // While it cannot be programmatically verified by the test runner,
  // it is extremely useful for developers running tests manually in a terminal
  // to visually verify characters are outputted correctly.

  // Print: "putwchar test: [a¢€𐍈]\n"
  constexpr wchar_t PREFIX[] = L"putwchar test: [";
  for (size_t i = 0; PREFIX[i] != L'\0'; ++i) {
    EXPECT_EQ(LIBC_NAMESPACE::putwchar(PREFIX[i]),
              static_cast<wint_t>(PREFIX[i]));
  }

  EXPECT_EQ(LIBC_NAMESPACE::putwchar(L'a'), static_cast<wint_t>(L'a'));
  EXPECT_EQ(LIBC_NAMESPACE::putwchar(L'¢'), static_cast<wint_t>(L'¢'));
#if WINT_MAX > 0xFFFF
  EXPECT_EQ(LIBC_NAMESPACE::putwchar(L'€'), static_cast<wint_t>(L'€'));
  EXPECT_EQ(LIBC_NAMESPACE::putwchar(L'𐍈'), static_cast<wint_t>(L'𐍈'));
#endif

  EXPECT_EQ(LIBC_NAMESPACE::putwchar(L']'), static_cast<wint_t>(L']'));
  EXPECT_EQ(LIBC_NAMESPACE::putwchar(L'\n'), static_cast<wint_t>(L'\n'));
}
