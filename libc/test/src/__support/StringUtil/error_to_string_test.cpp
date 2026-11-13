//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// ile
/// Unittests for get_error_string.
///
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/span.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/StringUtil/error_to_string.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::cpp::span;
using LIBC_NAMESPACE::cpp::string_view;

TEST(LlvmLibcErrorToStringTest, BuiltDescription) {
  char buffer[64];
  for (char &c : buffer)
    c = 'x';

  string_view result =
      LIBC_NAMESPACE::get_error_string(12345, span<char>(buffer, 64));
  ASSERT_TRUE(result == "Unknown error 12345");
}

TEST(LlvmLibcErrorToStringTest, BuiltCStringIsTerminated) {
  char buffer[64];
  for (char &c : buffer)
    c = 'x';

  ASSERT_STREQ(LIBC_NAMESPACE::get_error_cstring(12345, span<char>(buffer, 64)),
               "Unknown error 12345");
}

TEST(LlvmLibcErrorToStringTest, KnownCString) {
  ASSERT_STREQ(LIBC_NAMESPACE::get_error_cstring(0), "Success");
}

TEST(LlvmLibcErrorToStringTest, SmallBufferGivesTheBaseDescription) {
  char buffer[4];
  string_view result =
      LIBC_NAMESPACE::get_error_string(12345, span<char>(buffer, 4));
  ASSERT_TRUE(result == "Unknown error");
  ASSERT_STREQ(LIBC_NAMESPACE::get_error_cstring(12345, span<char>(buffer, 4)),
               "Unknown error");
}
