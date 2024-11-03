//===-- Unittests for stpncpy ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/span.h"
#include "src/string/stpncpy.h"
#include "test/UnitTest/Test.h"
#include <stddef.h> // For size_t.

class LlvmLibcStpncpyTest : public __llvm_libc::testing::Test {
public:
  void check_stpncpy(__llvm_libc::cpp::span<char> dst,
                     const __llvm_libc::cpp::span<const char> src, size_t n,
                     const __llvm_libc::cpp::span<const char> expected,
                     size_t expectedCopied) {
    // Making sure we don't overflow buffer.
    ASSERT_GE(dst.size(), n);
    // Making sure stpncpy returns a pointer to the end of dst.
    ASSERT_EQ(__llvm_libc::stpncpy(dst.data(), src.data(), n),
              dst.data() + expectedCopied);
    // Expected must be of the same size as dst.
    ASSERT_EQ(dst.size(), expected.size());
    // Expected and dst are the same.
    for (size_t i = 0; i < expected.size(); ++i)
      ASSERT_EQ(expected[i], dst[i]);
  }
};

TEST_F(LlvmLibcStpncpyTest, Untouched) {
  char dst[] = {'a', 'b'};
  const char src[] = {'x', '\0'};
  const char expected[] = {'a', 'b'};
  check_stpncpy(dst, src, 0, expected, 0);
}

TEST_F(LlvmLibcStpncpyTest, CopyOne) {
  char dst[] = {'a', 'b'};
  const char src[] = {'x', 'y'};
  const char expected[] = {'x', 'b'}; // no \0 is appended
  check_stpncpy(dst, src, 1, expected, 1);
}

TEST_F(LlvmLibcStpncpyTest, CopyNull) {
  char dst[] = {'a', 'b'};
  const char src[] = {'\0', 'y'};
  const char expected[] = {'\0', 'b'};
  check_stpncpy(dst, src, 1, expected, 0);
}

TEST_F(LlvmLibcStpncpyTest, CopyPastSrc) {
  char dst[] = {'a', 'b'};
  const char src[] = {'\0', 'y'};
  const char expected[] = {'\0', '\0'};
  check_stpncpy(dst, src, 2, expected, 0);
}

TEST_F(LlvmLibcStpncpyTest, CopyTwoNoNull) {
  char dst[] = {'a', 'b'};
  const char src[] = {'x', 'y'};
  const char expected[] = {'x', 'y'};
  check_stpncpy(dst, src, 2, expected, 2);
}

TEST_F(LlvmLibcStpncpyTest, CopyTwoWithNull) {
  char dst[] = {'a', 'b'};
  const char src[] = {'x', '\0'};
  const char expected[] = {'x', '\0'};
  check_stpncpy(dst, src, 2, expected, 1);
}
