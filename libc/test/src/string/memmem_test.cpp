//===-- Unittests for memmem ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/memmem.h"
#include "test/UnitTest/Test.h"

#include "src/string/string_utils.h"

namespace LIBC_NAMESPACE {

TEST(LlvmLibcMemmemTest, EmptyHaystackEmptyNeedleReturnsHaystck) {
  char *h = nullptr;
  char *n = nullptr;
  void *result = LIBC_NAMESPACE::memmem(h, 0, n, 0);
  ASSERT_EQ(static_cast<char *>(result), h);
}

TEST(LlvmLibcMemmemTest, EmptyHaystackNonEmptyNeedleReturnsNull) {
  char *h = nullptr;
  char n[] = {'a', 'b', 'c'};
  void *result = LIBC_NAMESPACE::memmem(h, 0, n, sizeof(n));
  ASSERT_EQ(result, static_cast<void *>(nullptr));
}

TEST(LlvmLibcMemmemTest, EmptyNeedleReturnsHaystack) {
  char h[] = {'a', 'b', 'c'};
  char *n = nullptr;
  void *result = LIBC_NAMESPACE::memmem(h, sizeof(h), n, 0);
  ASSERT_EQ(static_cast<char *>(result), h + 0);
}

TEST(LlvmLibcMemmemTest, ExactMatchReturnsHaystack) {
  char h[] = {'a', 'b', 'c'};
  char n[] = {'a', 'b', 'c'};
  void *result = LIBC_NAMESPACE::memmem(h, sizeof(h), n, sizeof(n));
  ASSERT_EQ(static_cast<char *>(result), h + 0);
}

TEST(LlvmLibcMemmemTest, ReturnFirstMatchOfNeedle) {
  char h[] = {'a', 'a', 'b', 'c'};
  char n[] = {'a'};
  void *result = LIBC_NAMESPACE::memmem(h, sizeof(h), n, sizeof(n));
  ASSERT_EQ(static_cast<char *>(result), h + 0);
}

TEST(LlvmLibcMemmemTest, ReturnFirstExactMatchOfNeedle) {
  {
    char h[] = {'a', 'b', 'a', 'c', 'a', 'a'};
    char n[] = {'a', 'a'};
    void *result = LIBC_NAMESPACE::memmem(h, sizeof(h), n, sizeof(n));
    ASSERT_EQ(static_cast<char *>(result), h + 4);
  }
  {
    char h[] = {'a', 'a', 'b', 'a', 'b', 'a'};
    char n[] = {'a', 'b', 'a'};
    void *result = LIBC_NAMESPACE::memmem(h, sizeof(h), n, sizeof(n));
    ASSERT_EQ(static_cast<char *>(result), h + 1);
  }
}

TEST(LlvmLibcMemmemTest, NullTerminatorDoesNotInterruptMatch) {
  char h[] = {'\0', 'a', 'b'};
  char n[] = {'a', 'b'};
  void *result = LIBC_NAMESPACE::memmem(h, sizeof(h), n, sizeof(n));
  ASSERT_EQ(static_cast<char *>(result), h + 1);
}

TEST(LlvmLibcMemmemTest, ReturnNullIfNoExactMatch) {
  {
    char h[] = {'a'};
    char n[] = {'a', 'a'};
    void *result = LIBC_NAMESPACE::memmem(h, sizeof(h), n, sizeof(n));
    ASSERT_EQ(result, static_cast<void *>(nullptr));
  }
  {
    char h[] = {'a', 'A'};
    char n[] = {'a', 'a'};
    void *result = LIBC_NAMESPACE::memmem(h, sizeof(h), n, sizeof(n));
    ASSERT_EQ(result, static_cast<void *>(nullptr));
  }
  {
    char h[] = {'a'};
    char n[] = {'a', '\0'};
    void *result = LIBC_NAMESPACE::memmem(h, sizeof(h), n, sizeof(n));
    ASSERT_EQ(result, static_cast<void *>(nullptr));
  }
  {
    char h[] = {'\0'};
    char n[] = {'\0', '\0'};
    void *result = LIBC_NAMESPACE::memmem(h, sizeof(h), n, sizeof(n));
    ASSERT_EQ(result, static_cast<void *>(nullptr));
  }
}

TEST(LlvmLibcMemmemTest, ReturnMatchOfSpecifiedNeedleLength) {
  {
    char h[] = {'a', 'b', 'c'};
    char n[] = {'x', 'y', 'z'};
    void *result = LIBC_NAMESPACE::memmem(h, sizeof(h), n, 0);
    ASSERT_EQ(static_cast<char *>(result), h + 0);
  }
  {
    char h[] = {'a', 'b', 'c'};
    char n[] = {'b', 'c', 'a'};
    void *result = LIBC_NAMESPACE::memmem(h, sizeof(h), n, 2);
    ASSERT_EQ(static_cast<char *>(result), h + 1);
  }
}

TEST(LlvmLibcMemmemTest, ReturnNullIfInadequateHaystackLength) {
  {
    char h[] = {'a', 'b', 'c'};
    char n[] = {'c'};
    void *result = LIBC_NAMESPACE::memmem(h, 2, n, sizeof(n));
    ASSERT_EQ(result, static_cast<void *>(nullptr));
  }
  {
    char h[] = {'a', 'b', 'c'};
    char n[] = {'a', 'b', 'c'};
    void *result = LIBC_NAMESPACE::memmem(h, 2, n, sizeof(n));
    ASSERT_EQ(result, static_cast<void *>(nullptr));
  }
}
} // namespace LIBC_NAMESPACE
