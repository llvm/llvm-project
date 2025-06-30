//===-- Unittests for mempcpy ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "hdr/signal_macros.h"
#include "src/string/mempcpy.h"
#include "test/UnitTest/Test.h"

// Since this function just calls out to memcpy, and memcpy has its own unit
// tests, it is assumed that memcpy works. These tests are just for the specific
// mempcpy behavior (returning the end of what was copied).
TEST(LlvmLibcMempcpyTest, Simple) {
  const char *src = "12345";
  char dest[10] = {};
  void *result = LIBC_NAMESPACE::mempcpy(dest, src, 6);
  ASSERT_EQ(static_cast<char *>(result), dest + 6);
  ASSERT_STREQ(src, dest);
}

TEST(LlvmLibcMempcpyTest, ZeroCount) {
  const char *src = "12345";
  char dest[10];
  void *result = LIBC_NAMESPACE::mempcpy(dest, src, 0);
  ASSERT_EQ(static_cast<char *>(result), dest + 0);
}

#if defined(LIBC_ADD_NULL_CHECKS) && !defined(LIBC_HAS_SANITIZER)

TEST(LlvmLibcMempcpyTest, CrashOnNullPtr) {
  ASSERT_DEATH([]() { LIBC_NAMESPACE::mempcpy(nullptr, nullptr, 1); },
               WITH_SIGNAL(-1));
}

#endif // defined(LIBC_ADD_NULL_CHECKS) && !defined(LIBC_HAS_SANITIZER)
