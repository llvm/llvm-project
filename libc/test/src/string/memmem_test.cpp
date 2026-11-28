//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for memmem.
///
//===----------------------------------------------------------------------===//

#include "src/__support/macros/config.h"
#include "src/string/memmem.h"
#include "test/UnitTest/Test.h"

#include "src/string/string_utils.h"

namespace LIBC_NAMESPACE_DECL {

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

TEST(LlvmLibcMemmemTest, PeriodicNeedles) {
  {
    const char h[] = "aaabaaaa";
    const char n[] = "aaaa";
    void *result = LIBC_NAMESPACE::memmem(h, sizeof(h) - 1, n, sizeof(n) - 1);
    ASSERT_EQ(static_cast<const char *>(result), h + 4);
  }
  {
    const char h[] = "bbababab";
    const char n[] = "ababab";
    void *result = LIBC_NAMESPACE::memmem(h, sizeof(h) - 1, n, sizeof(n) - 1);
    ASSERT_EQ(static_cast<const char *>(result), h + 2);
  }
  {
    const char h[] = "xabaaabaa";
    const char n[] = "aabaa";
    void *result = LIBC_NAMESPACE::memmem(h, sizeof(h) - 1, n, sizeof(n) - 1);
    ASSERT_EQ(static_cast<const char *>(result), h + 4);
  }
  {
    const char h[] = "xxabaaabaaab";
    const char n[] = "aaabaaab";
    void *result = LIBC_NAMESPACE::memmem(h, sizeof(h) - 1, n, sizeof(n) - 1);
    ASSERT_EQ(static_cast<const char *>(result), h + 4);
  }
  {
    const char h[] = "xxabaaabaaa";
    const char n[] = "aaabaaa";
    void *result = LIBC_NAMESPACE::memmem(h, sizeof(h) - 1, n, sizeof(n) - 1);
    ASSERT_EQ(static_cast<const char *>(result), h + 4);
  }
  {
    const char h[] = "aaabaaacaaabaaab";
    const char n[] = "aaabaaab";
    void *result = LIBC_NAMESPACE::memmem(h, sizeof(h) - 1, n, sizeof(n) - 1);
    ASSERT_EQ(static_cast<const char *>(result), h + 8);
  }
}

TEST(LlvmLibcMemmemTest, NonPeriodicNeedles) {
  {
    const char h[] = "xbabcababcab";
    const char n[] = "ababcab";
    void *result = LIBC_NAMESPACE::memmem(h, sizeof(h) - 1, n, sizeof(n) - 1);
    ASSERT_EQ(static_cast<const char *>(result), h + 5);
  }
  {
    const char h[] = "ababcac_ababcab";
    const char n[] = "ababcab";
    void *result = LIBC_NAMESPACE::memmem(h, sizeof(h) - 1, n, sizeof(n) - 1);
    ASSERT_EQ(static_cast<const char *>(result), h + 8);
  }
  {
    const char h[] = "aaaaaaaaab";
    const char n[] = "aaaab";
    void *result = LIBC_NAMESPACE::memmem(h, sizeof(h) - 1, n, sizeof(n) - 1);
    ASSERT_EQ(static_cast<const char *>(result), h + 5);
  }
  {
    const char h[] = "xaaaabaaaa";
    const char n[] = "baaaa";
    void *result = LIBC_NAMESPACE::memmem(h, sizeof(h) - 1, n, sizeof(n) - 1);
    ASSERT_EQ(static_cast<const char *>(result), h + 5);
  }
}

TEST(LlvmLibcMemmemTest, HighBitBytes) {
  const unsigned char h[] = {0x80, 0xff, 0x00, 0x7f, 0x80, 0xff, 0x81};
  const unsigned char n[] = {0x80, 0xff, 0x81};
  void *result = LIBC_NAMESPACE::memmem(h, sizeof(h), n, sizeof(n));
  ASSERT_EQ(static_cast<const unsigned char *>(result), h + 4);
}

TEST(LlvmLibcMemmemTest, RepetitiveLongInputs) {
  constexpr size_t HAYSTACK_LEN = 4096;
  constexpr size_t NEEDLE_LEN = 2048;
  char haystack[HAYSTACK_LEN];
  char needle[NEEDLE_LEN];

  for (size_t i = 0; i < HAYSTACK_LEN; ++i)
    haystack[i] = 'a';
  for (size_t i = 0; i < NEEDLE_LEN; ++i)
    needle[i] = 'a';

  needle[NEEDLE_LEN - 1] = 'b';
  ASSERT_EQ(LIBC_NAMESPACE::memmem(haystack, HAYSTACK_LEN, needle, NEEDLE_LEN),
            static_cast<void *>(nullptr));

  haystack[HAYSTACK_LEN - 1] = 'b';
  ASSERT_EQ(static_cast<char *>(LIBC_NAMESPACE::memmem(haystack, HAYSTACK_LEN,
                                                       needle, NEEDLE_LEN)),
            haystack + (HAYSTACK_LEN - NEEDLE_LEN));

  haystack[HAYSTACK_LEN - 1] = 'a';
  needle[NEEDLE_LEN - 1] = 'a';
  needle[0] = 'b';
  ASSERT_EQ(LIBC_NAMESPACE::memmem(haystack, HAYSTACK_LEN, needle, NEEDLE_LEN),
            static_cast<void *>(nullptr));

  haystack[HAYSTACK_LEN - NEEDLE_LEN] = 'b';
  ASSERT_EQ(static_cast<char *>(LIBC_NAMESPACE::memmem(haystack, HAYSTACK_LEN,
                                                       needle, NEEDLE_LEN)),
            haystack + (HAYSTACK_LEN - NEEDLE_LEN));
}
} // namespace LIBC_NAMESPACE_DECL
