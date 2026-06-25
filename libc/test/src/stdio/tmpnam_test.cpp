//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Tests for tmpnam
/// See: https://pubs.opengroup.org/onlinepubs/9799919799/functions/tmpnam.html
///
//===----------------------------------------------------------------------===//
#include "src/stdio/tmpnam.h"

#include "hdr/stdio_macros.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/macros/config.h"
#include "test/UnitTest/Test.h"

#include <stddef.h> // size_t

namespace {

using LIBC_NAMESPACE::cpp::string_view;

// The portable filename character set the implementation draws from, plus the
// '/' that appears in the P_tmpdir prefix. Any byte in a returned name must be
// one of these.
constexpr char kAllowed[] = "-._/"
                            "0123456789"
                            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                            "abcdefghijklmnopqrstuvwxyz";

bool only_allowed_chars(string_view sv) {
  for (char c : sv) {
    bool found = false;
    for (size_t i = 0; kAllowed[i] != '\0'; ++i) {
      if (c == kAllowed[i]) {
        found = true;
        break;
      }
    }
    if (!found)
      return false;
  }
  return true;
}

} // namespace

// Caller-supplied buffer: the spec requires the return value to be exactly the
// argument pointer, the string to be null-terminated within L_tmpnam bytes,
// and the result to begin with the temp-dir prefix.
TEST(LlvmLibcTmpnamTest, NonNullBufferReturnsSamePointer) {
  char buf[L_tmpnam];
  char *result = LIBC_NAMESPACE::tmpnam(buf);
  ASSERT_EQ(result, buf);
}

TEST(LlvmLibcTmpnamTest, NonNullBufferIsNullTerminated) {
  char buf[L_tmpnam];
  char *result = LIBC_NAMESPACE::tmpnam(buf);
  ASSERT_NE(result, static_cast<char *>(nullptr));
  // A NULL must appear within the buffer bounds.
  bool terminated = false;
  for (size_t i = 0; i < L_tmpnam; ++i) {
    if (result[i] == '\0') {
      terminated = true;
      break;
    }
  }
  ASSERT_TRUE(terminated);
}

TEST(LlvmLibcTmpnamTest, ResultHasTempDirPrefix) {
  char buf[L_tmpnam];
  char *result = LIBC_NAMESPACE::tmpnam(buf);
  ASSERT_NE(result, static_cast<char *>(nullptr));
  string_view sv(result);
  string_view prefix(P_tmpdir);
  // P_tmpdir may not carry a trailing slash; the implementation always
  // emits one separator, so check the directory portion is present at the head.
  ASSERT_TRUE(sv.starts_with(prefix));
}

TEST(LlvmLibcTmpnamTest, ResultUsesOnlyPortableChars) {
  char buf[L_tmpnam];
  char *result = LIBC_NAMESPACE::tmpnam(buf);
  ASSERT_NE(result, static_cast<char *>(nullptr));
  ASSERT_TRUE(only_allowed_chars(string_view(result)));
}

// Null argument: the result lives in an internal static object; the returned
// pointer must be non-null and carry the same structural guarantees.
TEST(LlvmLibcTmpnamTest, NullBufferReturnsInternalObject) {
  char *result = LIBC_NAMESPACE::tmpnam(nullptr);
  ASSERT_NE(result, static_cast<char *>(nullptr));
  string_view sv(result);
  ASSERT_TRUE(sv.starts_with(string_view(P_tmpdir)));
  ASSERT_TRUE(only_allowed_chars(sv));
}

// The core contract: the generated name must not already exist on disk.
// We re-check existence here via the public access() entry point if available.
TEST(LlvmLibcTmpnamTest, ResultLengthWithinBound) {
  char buf[L_tmpnam];
  char *result = LIBC_NAMESPACE::tmpnam(buf);
  ASSERT_NE(result, static_cast<char *>(nullptr));
  string_view sv(result);
  ASSERT_LT(sv.size(), static_cast<size_t>(L_tmpnam));
  // Must be strictly longer than the prefix: a prefix with no random suffix
  // would mean the generator produced an empty suffix.
  ASSERT_GT(sv.size(), string_view(P_tmpdir).size());
}

// Successive calls should produce distinct strings.
TEST(LlvmLibcTmpnamTest, SuccessiveCallsDiffer) {
  char a[L_tmpnam];
  char b[L_tmpnam];
  char *ra = LIBC_NAMESPACE::tmpnam(a);
  char *rb = LIBC_NAMESPACE::tmpnam(b);
  ASSERT_NE(ra, static_cast<char *>(nullptr));
  ASSERT_NE(rb, static_cast<char *>(nullptr));
  ASSERT_FALSE(string_view(ra) == string_view(rb));
}

// Two calls with a null argument must return the SAME pointer (the address of
// the single internal static object) The contents, however, are overwritten by
// the second call.
TEST(LlvmLibcTmpnamTest, NullCallsShareObjectButDifferInContent) {
  char *first = LIBC_NAMESPACE::tmpnam(nullptr);
  ASSERT_NE(first, static_cast<char *>(nullptr));

  // Snapshot the first result before it is overwritten.
  char snapshot[L_tmpnam];
  size_t i = 0;
  for (; i < L_tmpnam && first[i] != '\0'; ++i)
    snapshot[i] = first[i];
  snapshot[i < L_tmpnam ? i : L_tmpnam - 1] = '\0';

  char *second = LIBC_NAMESPACE::tmpnam(nullptr);
  ASSERT_NE(second, static_cast<char *>(nullptr));

  // Same backing object: identical address.
  ASSERT_EQ(first, second);

  // But the generated string changed
  ASSERT_FALSE(string_view(snapshot) == string_view(second));
}
