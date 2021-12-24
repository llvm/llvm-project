//===-- Unittests for fwrite ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/Array.h"
#include "src/stdio/FILE.h"
#include "src/stdio/fwrite.h"
#include "utils/UnitTest/Test.h"

TEST(LlvmLibcStdio, FWriteBasic) {
  struct StrcpyFile : __llvm_libc::FILE {
    char *buf;
  } f;
  char array[6];
  f.buf = array;
  f.write = +[](__llvm_libc::FILE *file, const char *ptr, size_t size) {
    StrcpyFile *strcpyFile = static_cast<StrcpyFile *>(file);
    for (size_t i = 0; i < size; ++i)
      strcpyFile->buf[i] = ptr[i];
    return size;
  };
  EXPECT_EQ(fwrite("hello", 1, 6, &f), 6UL);
  EXPECT_STREQ(array, "hello");
}
