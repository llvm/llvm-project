//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for putwc
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/stdint_proxy.h"
#include "hdr/wchar_macros.h" // For WEOF
#include "src/stdio/fclose.h"
#include "src/stdio/ferror.h"
#include "src/stdio/fopen.h"
#include "src/stdio/fread.h"
#include "src/wchar/fwide.h"
#include "src/wchar/putwc.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcPutwcTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcPutwcTest, WriteASCII) {
  const auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("putwc_ascii.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);

  // Initial orientation should be 0
  EXPECT_EQ(LIBC_NAMESPACE::fwide(file, 0), 0);

  EXPECT_EQ(LIBC_NAMESPACE::putwc(L'a', file), static_cast<wint_t>(L'a'));
  EXPECT_EQ(LIBC_NAMESPACE::putwc(L'b', file), static_cast<wint_t>(L'b'));
  EXPECT_EQ(LIBC_NAMESPACE::putwc(L'c', file), static_cast<wint_t>(L'c'));

  // Orientation should now be wide
  EXPECT_GT(LIBC_NAMESPACE::fwide(file, 0), 0);

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);

  // Read back raw bytes to verify
  file = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);

  unsigned char buffer[5] = {0};
  ASSERT_EQ(LIBC_NAMESPACE::fread(buffer, 1, 3, file), size_t(3));
  EXPECT_EQ(buffer[0], static_cast<unsigned char>('a'));
  EXPECT_EQ(buffer[1], static_cast<unsigned char>('b'));
  EXPECT_EQ(buffer[2], static_cast<unsigned char>('c'));

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}

TEST_F(LlvmLibcPutwcTest, WriteUtf8) {
  const auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("putwc_utf8.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);

  // a   -> L'a' (1-byte)
  // ¢   -> L'¢' (2-byte)
  // €   -> L'€' (3-byte)
  // 𐍈   -> L'𐍈' (4-byte)
  EXPECT_EQ(LIBC_NAMESPACE::putwc(L'a', file), static_cast<wint_t>(L'a'));
  EXPECT_EQ(LIBC_NAMESPACE::putwc(L'¢', file), static_cast<wint_t>(L'¢'));
#if WINT_MAX > 0xFFFF
  EXPECT_EQ(LIBC_NAMESPACE::putwc(L'€', file), static_cast<wint_t>(L'€'));
  EXPECT_EQ(LIBC_NAMESPACE::putwc(L'𐍈', file), static_cast<wint_t>(L'𐍈'));
  constexpr size_t EXPECTED_BYTES = 10;
#else
  constexpr size_t EXPECTED_BYTES = 6;
#endif

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);

  // Read back raw bytes and verify strict UTF-8 sequences on disk
  file = LIBC_NAMESPACE::fopen(FILENAME, "r");
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

TEST_F(LlvmLibcPutwcTest, InvalidStream) {
  const auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("putwc_invalid.test"));

  // Create file
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);
  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);

  // Open read-only
  file = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);

  ASSERT_ERRNO_SUCCESS();
  EXPECT_EQ(LIBC_NAMESPACE::putwc(L'a', file), static_cast<wint_t>(WEOF));
  ASSERT_ERRNO_EQ(EBADF);
  EXPECT_NE(LIBC_NAMESPACE::ferror(file), 0);

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}

TEST_F(LlvmLibcPutwcTest, ByteModeFailure) {
  const auto FILENAME =
      libc_make_test_file_path(APPEND_LIBC_TEST("putwc_bytemode.test"));
  ::FILE *file = LIBC_NAMESPACE::fopen(FILENAME, "w+");
  ASSERT_FALSE(file == nullptr);

  // Orient to byte mode
  EXPECT_LT(LIBC_NAMESPACE::fwide(file, -1), 0);

  // Writing wide char should fail and set errno to EINVAL
  EXPECT_EQ(LIBC_NAMESPACE::putwc(L'a', file), static_cast<wint_t>(WEOF));
  ASSERT_ERRNO_EQ(EINVAL);
  EXPECT_NE(LIBC_NAMESPACE::ferror(file), 0);

  ASSERT_EQ(LIBC_NAMESPACE::fclose(file), 0);
}
