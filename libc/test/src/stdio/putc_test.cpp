//===-- Unittests for putc ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fclose.h"
#include "src/stdio/ferror.h"
#include "src/stdio/fopen.h"
#include "src/stdio/fread.h"

#include "src/stdio/putc.h"

#include "test/UnitTest/Test.h"

#include <stdio.h>

TEST(LlvmLibcPutcTest, WriteToFile) {
  constexpr char FILENAME[] = "testdata/putc_output.test";
  ::FILE *file = __llvm_libc::fopen(FILENAME, "w");
  ASSERT_FALSE(file == nullptr);

  constexpr char simple[] = "simple letters";
  for (size_t i = 0; i < sizeof(simple); ++i) {
    ASSERT_EQ(__llvm_libc::putc(simple[i], file), 0);
  }

  ASSERT_EQ(0, __llvm_libc::fclose(file));

  file = __llvm_libc::fopen(FILENAME, "r");
  ASSERT_FALSE(file == nullptr);
  char data[50];

  ASSERT_EQ(__llvm_libc::fread(data, 1, sizeof(simple) - 1, file),
            sizeof(simple) - 1);
  data[sizeof(simple) - 1] = '\0';

  ASSERT_STREQ(data, simple);

  ASSERT_EQ(__llvm_libc::ferror(file), 0);
  EXPECT_LT(__llvm_libc::putc('L', file), 0);

  ASSERT_EQ(__llvm_libc::fclose(file), 0);
}
