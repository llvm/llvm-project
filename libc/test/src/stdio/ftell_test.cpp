//===-- Unittests for ftell -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fclose.h"
#include "src/stdio/fopen.h"
#include "src/stdio/fread.h"
#include "src/stdio/fseek.h"
#include "src/stdio/ftell.h"
#include "src/stdio/fwrite.h"
#include "src/stdio/setvbuf.h"
#include "test/UnitTest/Test.h"

#include <stdio.h>

class LlvmLibcFTellTest : public __llvm_libc::testing::Test {
protected:
  void test_with_bufmode(int bufmode) {
    constexpr char FILENAME[] = "testdata/ftell.test";
    // We will set a special buffer to the file so that we guarantee buffering.
    constexpr size_t BUFFER_SIZE = 1024;
    char buffer[BUFFER_SIZE];
    ::FILE *file = __llvm_libc::fopen(FILENAME, "w+");
    ASSERT_FALSE(file == nullptr);
    ASSERT_EQ(__llvm_libc::setvbuf(file, buffer, bufmode, BUFFER_SIZE), 0);

    // Include few '\n' chars to test when |bufmode| is _IOLBF.
    constexpr char CONTENT[] = "12\n345\n6789";
    constexpr size_t WRITE_SIZE = sizeof(CONTENT) - 1;
    ASSERT_EQ(WRITE_SIZE, __llvm_libc::fwrite(CONTENT, 1, WRITE_SIZE, file));
    // The above write should have buffered the written data and not have
    // trasferred it to the underlying stream. But, ftell operation should
    // still return the correct effective offset.
    ASSERT_EQ(size_t(__llvm_libc::ftell(file)), WRITE_SIZE);

    long offset = 5;
    ASSERT_EQ(0, __llvm_libc::fseek(file, offset, SEEK_SET));
    ASSERT_EQ(__llvm_libc::ftell(file), offset);
    ASSERT_EQ(0, __llvm_libc::fseek(file, -offset, SEEK_END));
    ASSERT_EQ(size_t(__llvm_libc::ftell(file)), size_t(WRITE_SIZE - offset));

    ASSERT_EQ(0, __llvm_libc::fseek(file, 0, SEEK_SET));
    constexpr size_t READ_SIZE = WRITE_SIZE / 2;
    char data[READ_SIZE];
    // Reading a small amount will actually read out much more data and
    // buffer it. But, ftell should return the correct effective offset.
    ASSERT_EQ(READ_SIZE, __llvm_libc::fread(data, 1, READ_SIZE, file));
    ASSERT_EQ(size_t(__llvm_libc::ftell(file)), READ_SIZE);

    ASSERT_EQ(0, __llvm_libc::fclose(file));
  }
};

TEST_F(LlvmLibcFTellTest, TellWithFBF) { test_with_bufmode(_IOFBF); }

TEST_F(LlvmLibcFTellTest, TellWithNBF) { test_with_bufmode(_IONBF); }

TEST_F(LlvmLibcFTellTest, TellWithLBF) { test_with_bufmode(_IOLBF); }
