//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for tmpnam
///
//===----------------------------------------------------------------------===//
#include "src/stdio/tmpnam.h"

#include "hdr/stdio_macros.h" // For L_tmpnam
#include "hdr/types/size_t.h"
#include "src/__support/CPP/string.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/macros/config.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::cpp::string;
using LIBC_NAMESPACE::cpp::string_view;

namespace {

constexpr string_view TMPDIR = "/tmp/";

// POSIX portable filename character set.
constexpr string_view ALLOWED = "0123456789"
                                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                "abcdefghijklmnopqrstuvwxyz"
                                "_.";

bool only_allowed_chars(string_view sv) {
  for (char c : sv) {
    if (!ALLOWED.contains(c))
      return false;
  }
  return true;
}

} // namespace

// For caller-supplied buffer the return value should be
// the argument pointer
TEST(LlvmLibcTmpnamTest, NonNullBufferReturnsSamePointer) {
  char buf[L_tmpnam];
  char *result = LIBC_NAMESPACE::tmpnam(buf);
  ASSERT_EQ(result, buf);
}

TEST(LlvmLibcTmpnamTest, NonNullBufferIsNullTerminated) {
  char buf[L_tmpnam];
  char *result = LIBC_NAMESPACE::tmpnam(buf);
  ASSERT_NE(result, static_cast<char *>(nullptr));
  string_view sv(result);
  ASSERT_EQ(sv.size(), static_cast<size_t>(L_tmpnam - 1));
  ASSERT_EQ(sv[L_tmpnam - 1], '\0');
}

TEST(LlvmLibcTmpnamTest, ResultUsesOnlyPortableChars) {
  char buf[L_tmpnam];
  char *result = LIBC_NAMESPACE::tmpnam(buf);
  ASSERT_NE(result, static_cast<char *>(nullptr));
  string_view sv(result);
  ASSERT_TRUE(sv.starts_with(TMPDIR));
  ASSERT_TRUE(only_allowed_chars(sv.substr(TMPDIR.size())));
}

// For null argument, the result lives in an internal static object.
TEST(LlvmLibcTmpnamTest, NullBufferReturnsInternalObject) {
  char *result = LIBC_NAMESPACE::tmpnam(nullptr);
  ASSERT_NE(result, static_cast<char *>(nullptr));
  string_view sv(result);
  ASSERT_TRUE(sv.starts_with(TMPDIR));
  ASSERT_TRUE(only_allowed_chars(sv.substr(TMPDIR.size())));
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

// Two calls with a null argument must return the same
// pointer. The contents, however, are overwritten by
// the second call.
TEST(LlvmLibcTmpnamTest, NullCallsShareObjectButDifferInContent) {
  char *first = LIBC_NAMESPACE::tmpnam(nullptr);
  ASSERT_NE(first, static_cast<char *>(nullptr));

  // Snapshot the first result before it is overwritten.
  string snapshot(first);

  char *second = LIBC_NAMESPACE::tmpnam(nullptr);
  ASSERT_NE(second, static_cast<char *>(nullptr));
  ASSERT_EQ(first, second);

  string secondstr(second);
  // But the generated string changed
  ASSERT_FALSE(snapshot == secondstr);
}
