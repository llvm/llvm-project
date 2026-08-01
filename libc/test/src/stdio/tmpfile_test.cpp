//===-- Unittests for tmpfile --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
#include "hdr/stdio_macros.h"
#include "src/stdio/tmpfile.h"
#include "src/stdio/fclose.h"
#include "src/stdio/fflush.h"
#include "src/stdio/fread.h"
#include "src/stdio/fseek.h"
#include "src/stdio/ftell.h"
#include "src/stdio/fwrite.h"
#include "test/UnitTest/Test.h"

constexpr char TEST_STR[] = "test\xaa\t\xbb";
constexpr size_t TEST_STR_SIZE = sizeof(TEST_STR);

TEST(LlvmLibcTmpfileTest, CreationAndInvariants) {
  auto *file = LIBC_NAMESPACE::tmpfile();
  ASSERT_NE(file, nullptr);

  EXPECT_EQ(LIBC_NAMESPACE::ftell(file), 0L);

  size_t nbytes_written = LIBC_NAMESPACE::fwrite(TEST_STR, 1, TEST_STR_SIZE, file);
  EXPECT_EQ(nbytes_written, TEST_STR_SIZE);

  EXPECT_EQ(LIBC_NAMESPACE::fflush(file), 0);
  EXPECT_EQ(static_cast<size_t>(LIBC_NAMESPACE::ftell(file)), nbytes_written);

  EXPECT_EQ(LIBC_NAMESPACE::fseek(file, 0, SEEK_SET), 0);
  EXPECT_EQ(LIBC_NAMESPACE::ftell(file), 0L);


  char buff[TEST_STR_SIZE * 2];
  size_t nbytes_read = LIBC_NAMESPACE::fread(buff, 1, TEST_STR_SIZE, file);
  EXPECT_EQ(nbytes_read, TEST_STR_SIZE);

}

