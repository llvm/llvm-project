//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for FieldTokenizer.
///
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/span.h"
#include "src/pwd/field_tokenizer.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcFieldTokenizerTest, StandardPasswdLine) {
  char line[] = "root:x:0:0:root:/root:/bin/bash";
  LIBC_NAMESPACE::pwd::FieldTokenizer tokenizer(
      LIBC_NAMESPACE::cpp::span<char>(line, sizeof(line)));

  auto f1 = tokenizer.next_field();
  ASSERT_TRUE(f1.has_value());
  ASSERT_STREQ(f1->data(), "root");

  auto f2 = tokenizer.next_field();
  ASSERT_TRUE(f2.has_value());
  ASSERT_STREQ(f2->data(), "x");

  auto f3 = tokenizer.next_field();
  ASSERT_TRUE(f3.has_value());
  ASSERT_STREQ(f3->data(), "0");

  auto f4 = tokenizer.next_field();
  ASSERT_TRUE(f4.has_value());
  ASSERT_STREQ(f4->data(), "0");

  auto f5 = tokenizer.next_field();
  ASSERT_TRUE(f5.has_value());
  ASSERT_STREQ(f5->data(), "root");

  auto f6 = tokenizer.next_field();
  ASSERT_TRUE(f6.has_value());
  ASSERT_STREQ(f6->data(), "/root");

  auto f7 = tokenizer.next_field();
  ASSERT_TRUE(f7.has_value());
  ASSERT_STREQ(f7->data(), "/bin/bash");

  auto f8 = tokenizer.next_field();
  ASSERT_FALSE(f8.has_value());
}

TEST(LlvmLibcFieldTokenizerTest, EmptyFields) {
  char line[] = "a::c:";
  LIBC_NAMESPACE::pwd::FieldTokenizer tokenizer(
      LIBC_NAMESPACE::cpp::span<char>(line, sizeof(line)));

  auto f1 = tokenizer.next_field();
  ASSERT_TRUE(f1.has_value());
  ASSERT_STREQ(f1->data(), "a");

  auto f2 = tokenizer.next_field();
  ASSERT_TRUE(f2.has_value());
  ASSERT_STREQ(f2->data(), "");

  auto f3 = tokenizer.next_field();
  ASSERT_TRUE(f3.has_value());
  ASSERT_STREQ(f3->data(), "c");

  auto f4 = tokenizer.next_field();
  ASSERT_TRUE(f4.has_value());
  ASSERT_STREQ(f4->data(), "");

  auto f5 = tokenizer.next_field();
  ASSERT_FALSE(f5.has_value());
}

TEST(LlvmLibcFieldTokenizerTest, LeadingAndConsecutiveSeparators) {
  char line[] = ":first::last";
  LIBC_NAMESPACE::pwd::FieldTokenizer tokenizer(
      LIBC_NAMESPACE::cpp::span<char>(line, sizeof(line)));

  auto f1 = tokenizer.next_field();
  ASSERT_TRUE(f1.has_value());
  ASSERT_STREQ(f1->data(), "");

  auto f2 = tokenizer.next_field();
  ASSERT_TRUE(f2.has_value());
  ASSERT_STREQ(f2->data(), "first");

  auto f3 = tokenizer.next_field();
  ASSERT_TRUE(f3.has_value());
  ASSERT_STREQ(f3->data(), "");

  auto f4 = tokenizer.next_field();
  ASSERT_TRUE(f4.has_value());
  ASSERT_STREQ(f4->data(), "last");

  auto f5 = tokenizer.next_field();
  ASSERT_FALSE(f5.has_value());
}

TEST(LlvmLibcFieldTokenizerTest, SingleField) {
  char line[] = "single";
  LIBC_NAMESPACE::pwd::FieldTokenizer tokenizer(
      LIBC_NAMESPACE::cpp::span<char>(line, sizeof(line)));

  auto f1 = tokenizer.next_field();
  ASSERT_TRUE(f1.has_value());
  ASSERT_STREQ(f1->data(), "single");

  auto f2 = tokenizer.next_field();
  ASSERT_FALSE(f2.has_value());
}

TEST(LlvmLibcFieldTokenizerTest, EmptyBuffer) {
  char line[] = "";
  LIBC_NAMESPACE::pwd::FieldTokenizer tokenizer(
      LIBC_NAMESPACE::cpp::span<char>(line, sizeof(line)));

  auto f1 = tokenizer.next_field();
  ASSERT_TRUE(f1.has_value());
  ASSERT_STREQ(f1->data(), "");

  auto f2 = tokenizer.next_field();
  ASSERT_FALSE(f2.has_value());
}

TEST(LlvmLibcFieldTokenizerTest, CustomSeparator) {
  char line[] = "foo,bar,baz";
  LIBC_NAMESPACE::pwd::FieldTokenizer tokenizer(
      LIBC_NAMESPACE::cpp::span<char>(line, sizeof(line)), ',');

  auto f1 = tokenizer.next_field();
  ASSERT_TRUE(f1.has_value());
  ASSERT_STREQ(f1->data(), "foo");

  auto f2 = tokenizer.next_field();
  ASSERT_TRUE(f2.has_value());
  ASSERT_STREQ(f2->data(), "bar");

  auto f3 = tokenizer.next_field();
  ASSERT_TRUE(f3.has_value());
  ASSERT_STREQ(f3->data(), "baz");

  auto f4 = tokenizer.next_field();
  ASSERT_FALSE(f4.has_value());
}
