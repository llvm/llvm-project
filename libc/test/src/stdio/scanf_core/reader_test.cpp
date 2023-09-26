//===-- Unittests for the scanf String Reader -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/string_view.h"
#include "src/stdio/scanf_core/reader.h"

#include "test/UnitTest/Test.h"

TEST(LlvmLibcScanfStringReaderTest, Constructor) {
  char str[10];
  // buff_len justneeds to be a big number. The specific value isn't important
  // in the real world.
  LIBC_NAMESPACE::scanf_core::ReadBuffer rb{const_cast<char *>(str), 1000000};
  LIBC_NAMESPACE::scanf_core::Reader reader(&rb);
}

TEST(LlvmLibcScanfStringReaderTest, SimpleRead) {
  const char *str = "abc";
  LIBC_NAMESPACE::scanf_core::ReadBuffer rb{const_cast<char *>(str), 1000000};
  LIBC_NAMESPACE::scanf_core::Reader reader(&rb);

  for (size_t i = 0; i < sizeof("abc"); ++i) {
    ASSERT_EQ(str[i], reader.getc());
  }
}

TEST(LlvmLibcScanfStringReaderTest, ReadAndReverse) {
  const char *str = "abcDEF123";
  LIBC_NAMESPACE::scanf_core::ReadBuffer rb{const_cast<char *>(str), 1000000};
  LIBC_NAMESPACE::scanf_core::Reader reader(&rb);

  for (size_t i = 0; i < 5; ++i) {
    ASSERT_EQ(str[i], reader.getc());
  }

  // Move back by 3, cursor should now be on 2
  reader.ungetc(str[4]);
  reader.ungetc(str[3]);
  reader.ungetc(str[2]);

  for (size_t i = 2; i < 7; ++i) {
    ASSERT_EQ(str[i], reader.getc());
  }

  // Move back by 2, cursor should now be on 5
  reader.ungetc(str[6]);
  reader.ungetc(str[5]);

  for (size_t i = 5; i < 10; ++i) {
    ASSERT_EQ(str[i], reader.getc());
  }

  // Move back by 10, which should be back to the start.
  for (size_t i = 0; i < 10; ++i) {
    reader.ungetc(str[9 - i]);
  }

  // Check the whole string.
  for (size_t i = 0; i < sizeof("abcDEF123"); ++i) {
    ASSERT_EQ(str[i], reader.getc());
  }
}
