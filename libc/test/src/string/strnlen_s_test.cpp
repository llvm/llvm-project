//===-- Unittests for strnlen_s -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strnlen_s.h"
#include "test/UnitTest/Test.h"
#include <stddef.h>

TEST(LlvmLibcStrNLenSTest, NullPointerInput) {
  const char *str = nullptr;
  // If the string input is a null pointer, it should return 0 regardless of
  // the max len arg value.
  ASSERT_EQ(LIBC_NAMESPACE::strnlen_s(str, 0), size_t(0));
  ASSERT_EQ(LIBC_NAMESPACE::strnlen_s(str, 1), size_t(0));
}

// The semantics when the string input is not null are the same as strnlen. The
// following tests are copied from the latter's tests.

TEST(LlvmLibcStrNLenSTest, EmptyString) {
  const char *empty = "";
  ASSERT_EQ(static_cast<size_t>(0), LIBC_NAMESPACE::strnlen_s(empty, 0));
  // If N is greater than string length, this should still return 0.
  ASSERT_EQ(static_cast<size_t>(0), LIBC_NAMESPACE::strnlen_s(empty, 1));
}

TEST(LlvmLibcStrNLenSTest, OneCharacterString) {
  const char *single = "X";
  ASSERT_EQ(static_cast<size_t>(1), LIBC_NAMESPACE::strnlen_s(single, 1));
  // If N is zero, this should return 0.
  ASSERT_EQ(static_cast<size_t>(0), LIBC_NAMESPACE::strnlen_s(single, 0));
  // If N is greater than string length, this should still return 1.
  ASSERT_EQ(static_cast<size_t>(1), LIBC_NAMESPACE::strnlen_s(single, 2));
}

TEST(LlvmLibcStrNLenSTest, ManyCharacterString) {
  const char *many = "123456789";
  ASSERT_EQ(static_cast<size_t>(9), LIBC_NAMESPACE::strnlen_s(many, 9));
  // If N is smaller than the string length, it should return N.
  ASSERT_EQ(static_cast<size_t>(3), LIBC_NAMESPACE::strnlen_s(many, 3));
  // If N is zero, this should return 0.
  ASSERT_EQ(static_cast<size_t>(0), LIBC_NAMESPACE::strnlen_s(many, 0));
  // If N is greater than the string length, this should still return 9.
  ASSERT_EQ(static_cast<size_t>(9), LIBC_NAMESPACE::strnlen_s(many, 42));
}

TEST(LlvmLibcStrNLenSTest, CharactersAfterNullTerminatorShouldNotBeIncluded) {
  const char str[5] = {'a', 'b', 'c', '\0', 'd'};
  ASSERT_EQ(static_cast<size_t>(3), LIBC_NAMESPACE::strnlen_s(str, 3));
  // This should only read up to the null terminator.
  ASSERT_EQ(static_cast<size_t>(3), LIBC_NAMESPACE::strnlen_s(str, 4));
  ASSERT_EQ(static_cast<size_t>(3), LIBC_NAMESPACE::strnlen_s(str, 5));
}
