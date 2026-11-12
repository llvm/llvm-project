
//===-- Unittests for swab ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/swab.h"

#include "src/string/string_utils.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcSwabTest, NegativeSizeIsNoOp) {
  const char *from = "abc";
  char to[4] = {'x', 'y', 'z', '\0'};
  LIBC_NAMESPACE::swab(from, to, -1);
  ASSERT_STREQ(to, "xyz");
}

TEST(LlvmLibcSwabTest, ZeroSizeIsNoOp) {
  const char *from = "abc";
  char to[4] = {'x', 'y', 'z', '\0'};
  LIBC_NAMESPACE::swab(from, to, 0);
  ASSERT_STREQ(to, "xyz");
}

TEST(LlvmLibcSwabTest, SingleByteIsNoOp) {
  char from[] = {'a'};
  char to[4] = {'x', 'y', 'z', '\0'};
  LIBC_NAMESPACE::swab(from, to, sizeof(from));
  ASSERT_STREQ(to, "xyz");
}

TEST(LlvmLibcSwabTest, NullPtrsAreNotDeRefedIfNIsLessThanTwo) {
  // This test passes if a crash does not happen
  LIBC_NAMESPACE::swab(nullptr, nullptr, -1);
  LIBC_NAMESPACE::swab(nullptr, nullptr, 0);
  LIBC_NAMESPACE::swab(nullptr, nullptr, 1);
}

TEST(LlvmLibcSwabTest, BytesAreSwappedWithEvenN) {
  {
    const char *from = "ab";
    char to[3] = {};
    LIBC_NAMESPACE::swab(from, to,
                         LIBC_NAMESPACE::internal::string_length(from));
    ASSERT_STREQ(to, "ba");
  }
  {
    const char *from = "abcd";
    char to[5] = {};
    LIBC_NAMESPACE::swab(from, to,
                         LIBC_NAMESPACE::internal::string_length(from));
    ASSERT_STREQ(to, "badc");
  }
  {
    const char *from = "aAaAaA";
    char to[7] = {};
    LIBC_NAMESPACE::swab(from, to,
                         LIBC_NAMESPACE::internal::string_length(from));
    ASSERT_STREQ(to, "AaAaAa");
  }
}

TEST(LlvmLibcSwabTest, LastByteIgnoredWithOddN) {
  {
    const char *from = "aba";
    char to[3] = {};
    LIBC_NAMESPACE::swab(from, to,
                         LIBC_NAMESPACE::internal::string_length(from));
    ASSERT_STREQ(to, "ba");
  }
  {
    const char *from = "abcde";
    char to[5] = {};
    LIBC_NAMESPACE::swab(from, to,
                         LIBC_NAMESPACE::internal::string_length(from));
    ASSERT_STREQ(to, "badc");
  }
  {
    const char *from = "aAaAaAx";
    char to[7] = {};
    LIBC_NAMESPACE::swab(from, to,
                         LIBC_NAMESPACE::internal::string_length(from));
    ASSERT_STREQ(to, "AaAaAa");
  }
}
