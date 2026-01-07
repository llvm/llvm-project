//===-- Unittests for wcs_to_integer --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/str_to_integer.h"
#include <stddef.h>

#include "test/UnitTest/Test.h"

// This file is for testing the src_len argument and other internal interface
// features. Primary testing is done through the public interface.

TEST(LlvmLibcWcsToIntegerTest, SimpleLength) {
  auto result = LIBC_NAMESPACE::internal::strtointeger<int>(L"12345", 10, 10);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(5));
  ASSERT_EQ(result.value, 12345);

  result = LIBC_NAMESPACE::internal::strtointeger<int>(L"12345", 10, 2);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(2));
  ASSERT_EQ(result.value, 12);

  result = LIBC_NAMESPACE::internal::strtointeger<int>(L"12345", 10, 0);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(0));
  ASSERT_EQ(result.value, 0);
}

TEST(LlvmLibcWcsToIntegerTest, LeadingSpaces) {
  auto result =
      LIBC_NAMESPACE::internal::strtointeger<int>(L"     12345", 10, 15);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(10));
  ASSERT_EQ(result.value, 12345);

  result = LIBC_NAMESPACE::internal::strtointeger<int>(L"     12345", 10, 10);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(10));
  ASSERT_EQ(result.value, 12345);

  result = LIBC_NAMESPACE::internal::strtointeger<int>(L"     12345", 10, 7);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(7));
  ASSERT_EQ(result.value, 12);

  // Use a non-null-terminated buffer to test for possible OOB access.
  wchar_t buf[5] = {L' ', L' ', L' ', L' ', L' '};
  result = LIBC_NAMESPACE::internal::strtointeger<int>(buf, 10, 5);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(0));
  ASSERT_EQ(result.value, 0);

  result = LIBC_NAMESPACE::internal::strtointeger<int>(buf, 10, 0);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(0));
  ASSERT_EQ(result.value, 0);
}

TEST(LlvmLibcWcsToIntegerTest, LeadingSign) {
  auto result = LIBC_NAMESPACE::internal::strtointeger<int>(L"+12345", 10, 10);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(6));
  ASSERT_EQ(result.value, 12345);

  result = LIBC_NAMESPACE::internal::strtointeger<int>(L"-12345", 10, 10);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(6));
  ASSERT_EQ(result.value, -12345);

  result = LIBC_NAMESPACE::internal::strtointeger<int>(L"+12345", 10, 6);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(6));
  ASSERT_EQ(result.value, 12345);

  result = LIBC_NAMESPACE::internal::strtointeger<int>(L"-12345", 10, 6);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(6));
  ASSERT_EQ(result.value, -12345);

  result = LIBC_NAMESPACE::internal::strtointeger<int>(L"+12345", 10, 3);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(3));
  ASSERT_EQ(result.value, 12);

  result = LIBC_NAMESPACE::internal::strtointeger<int>(L"-12345", 10, 3);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(3));
  ASSERT_EQ(result.value, -12);

  result = LIBC_NAMESPACE::internal::strtointeger<int>(L"+12345", 10, 1);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(0));
  ASSERT_EQ(result.value, 0);

  result = LIBC_NAMESPACE::internal::strtointeger<int>(L"-12345", 10, 1);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(0));
  ASSERT_EQ(result.value, 0);

  result = LIBC_NAMESPACE::internal::strtointeger<int>(L"+12345", 10, 0);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(0));
  ASSERT_EQ(result.value, 0);

  result = LIBC_NAMESPACE::internal::strtointeger<int>(L"-12345", 10, 0);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(0));
  ASSERT_EQ(result.value, 0);
}

TEST(LlvmLibcWcsToIntegerTest, Base16PrefixAutoSelect) {
  auto result = LIBC_NAMESPACE::internal::strtointeger<int>(L"0x12345", 0, 10);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(7));
  ASSERT_EQ(result.value, 0x12345);

  result = LIBC_NAMESPACE::internal::strtointeger<int>(L"0x12345", 0, 7);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(7));
  ASSERT_EQ(result.value, 0x12345);

  result = LIBC_NAMESPACE::internal::strtointeger<int>(L"0x12345", 0, 5);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(5));
  ASSERT_EQ(result.value, 0x123);

  result = LIBC_NAMESPACE::internal::strtointeger<int>(L"0x12345", 0, 2);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(1));
  ASSERT_EQ(result.value, 0);

  result = LIBC_NAMESPACE::internal::strtointeger<int>(L"0x12345", 0, 0);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(0));
  ASSERT_EQ(result.value, 0);
}

TEST(LlvmLibcWcsToIntegerTest, Base16PrefixManualSelect) {
  auto result = LIBC_NAMESPACE::internal::strtointeger<int>(L"0x12345", 16, 10);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(7));
  ASSERT_EQ(result.value, 0x12345);

  result = LIBC_NAMESPACE::internal::strtointeger<int>(L"0x12345", 16, 7);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(7));
  ASSERT_EQ(result.value, 0x12345);

  result = LIBC_NAMESPACE::internal::strtointeger<int>(L"0x12345", 16, 5);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(5));
  ASSERT_EQ(result.value, 0x123);

  result = LIBC_NAMESPACE::internal::strtointeger<int>(L"0x12345", 16, 2);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(1));
  ASSERT_EQ(result.value, 0);

  result = LIBC_NAMESPACE::internal::strtointeger<int>(L"0x12345", 16, 0);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(0));
  ASSERT_EQ(result.value, 0);
}

TEST(LlvmLibcWcsToIntegerTest, Base8PrefixAutoSelect) {
  auto result = LIBC_NAMESPACE::internal::strtointeger<int>(L"012345", 0, 10);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(6));
  ASSERT_EQ(result.value, 012345);

  result = LIBC_NAMESPACE::internal::strtointeger<int>(L"012345", 0, 6);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(6));
  ASSERT_EQ(result.value, 012345);

  result = LIBC_NAMESPACE::internal::strtointeger<int>(L"012345", 0, 4);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(4));
  ASSERT_EQ(result.value, 0123);

  result = LIBC_NAMESPACE::internal::strtointeger<int>(L"012345", 0, 1);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(1));
  ASSERT_EQ(result.value, 0);

  result = LIBC_NAMESPACE::internal::strtointeger<int>(L"012345", 0, 0);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(0));
  ASSERT_EQ(result.value, 0);
}

TEST(LlvmLibcWcsToIntegerTest, Base8PrefixManualSelect) {
  auto result = LIBC_NAMESPACE::internal::strtointeger<int>(L"012345", 8, 10);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(6));
  ASSERT_EQ(result.value, 012345);

  result = LIBC_NAMESPACE::internal::strtointeger<int>(L"012345", 8, 6);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(6));
  ASSERT_EQ(result.value, 012345);

  result = LIBC_NAMESPACE::internal::strtointeger<int>(L"012345", 8, 4);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(4));
  ASSERT_EQ(result.value, 0123);

  result = LIBC_NAMESPACE::internal::strtointeger<int>(L"012345", 8, 1);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(1));
  ASSERT_EQ(result.value, 0);

  result = LIBC_NAMESPACE::internal::strtointeger<int>(L"012345", 8, 0);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(0));
  ASSERT_EQ(result.value, 0);
}

TEST(LlvmLibcWcsToIntegerTest, CombinedTests) {
  auto result =
      LIBC_NAMESPACE::internal::strtointeger<int>(L"    -0x123", 0, 10);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(10));
  ASSERT_EQ(result.value, -0x123);

  result = LIBC_NAMESPACE::internal::strtointeger<int>(L"    -0x123", 0, 8);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(8));
  ASSERT_EQ(result.value, -0x1);

  result = LIBC_NAMESPACE::internal::strtointeger<int>(L"    -0x123", 0, 7);
  EXPECT_FALSE(result.has_error());
  EXPECT_EQ(result.parsed_len, ptrdiff_t(6));
  ASSERT_EQ(result.value, 0);
}
