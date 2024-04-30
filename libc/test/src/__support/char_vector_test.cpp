//===-- Unittests for char_vector ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/char_vector.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::CharVector;

TEST(LlvmLibcCharVectorTest, InitializeCheck) {
  CharVector v;
  ASSERT_EQ(v.length(), size_t(0));
}

TEST(LlvmLibcCharVectorTest, AppendShort) {
  CharVector v;
  ASSERT_EQ(v.length(), size_t(0));

  static constexpr char test_str[] = "1234567890";
  for (size_t i = 0; test_str[i] != '\0'; ++i) {
    v.append(test_str[i]);
  }
  ASSERT_STREQ(v.c_str(), test_str);
}

TEST(LlvmLibcCharVectorTest, AppendMedium) {
  CharVector v;
  ASSERT_EQ(v.length(), size_t(0));

  // 100 characters (each row is 50)
  static constexpr char test_str[] =
      "12345678901234567890123456789012345678901234567890"
      "ABCDEFGHIJKLMNOPQRSTUVWXYabcdefghijklmnopqrstuvwxy";
  for (size_t i = 0; test_str[i] != '\0'; ++i) {
    ASSERT_EQ(v.length(), i);
    v.append(test_str[i]);
  }
  ASSERT_STREQ(v.c_str(), test_str);
  ASSERT_EQ(v.length(), size_t(100));
}

TEST(LlvmLibcCharVectorTest, AppendLong) {
  CharVector v;
  ASSERT_EQ(v.length(), size_t(0));

  // 1000 characters
  static constexpr char test_str[] =
      "12345678901234567890123456789012345678901234567890"
      "ABCDEFGHIJKLMNOPQRSTUVWXYabcdefghijklmnopqrstuvwxy"
      "12345678901234567890123456789012345678901234567890"
      "ABCDEFGHIJKLMNOPQRSTUVWXYabcdefghijklmnopqrstuvwxy"
      "12345678901234567890123456789012345678901234567890"
      "ABCDEFGHIJKLMNOPQRSTUVWXYabcdefghijklmnopqrstuvwxy"
      "12345678901234567890123456789012345678901234567890"
      "ABCDEFGHIJKLMNOPQRSTUVWXYabcdefghijklmnopqrstuvwxy"
      "12345678901234567890123456789012345678901234567890"
      "ABCDEFGHIJKLMNOPQRSTUVWXYabcdefghijklmnopqrstuvwxy"
      "12345678901234567890123456789012345678901234567890"
      "ABCDEFGHIJKLMNOPQRSTUVWXYabcdefghijklmnopqrstuvwxy"
      "12345678901234567890123456789012345678901234567890"
      "ABCDEFGHIJKLMNOPQRSTUVWXYabcdefghijklmnopqrstuvwxy"
      "12345678901234567890123456789012345678901234567890"
      "ABCDEFGHIJKLMNOPQRSTUVWXYabcdefghijklmnopqrstuvwxy"
      "12345678901234567890123456789012345678901234567890"
      "ABCDEFGHIJKLMNOPQRSTUVWXYabcdefghijklmnopqrstuvwxy"
      "12345678901234567890123456789012345678901234567890"
      "ABCDEFGHIJKLMNOPQRSTUVWXYabcdefghijklmnopqrstuvwxy";
  for (size_t i = 0; test_str[i] != '\0'; ++i) {
    ASSERT_EQ(v.length(), i);
    ASSERT_TRUE(v.append(test_str[i]));
  }
  ASSERT_EQ(v.length(), size_t(1000));
  ASSERT_STREQ(v.c_str(), test_str);
}
