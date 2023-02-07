//===-- Unittests for strerror --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strerror_r.h"
#include "test/UnitTest/Test.h"

#include <stddef.h>

// This tests the gnu variant of strerror_r (which returns a char*).
TEST(LlvmLibcStrErrorRTest, GnuVariantTests) {
  const size_t BUFF_SIZE = 128;
  char buffer[BUFF_SIZE];
  buffer[0] = '\0';
  // If strerror_r returns a constant string, then it shouldn't affect the
  // buffer.
  ASSERT_STREQ(__llvm_libc::strerror_r(0, buffer, BUFF_SIZE), "Success");
  ASSERT_EQ(buffer[0], '\0');

  // Else it should write the result to the provided buffer.
  ASSERT_STREQ(__llvm_libc::strerror_r(-1, buffer, BUFF_SIZE),
               "Unknown error -1");
  ASSERT_STREQ(buffer, "Unknown error -1");
}
