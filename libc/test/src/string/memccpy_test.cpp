//===-- Unittests for memccpy ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/ArrayRef.h"
#include "src/string/memccpy.h"
#include "utils/UnitTest/Test.h"
#include <stddef.h> // For size_t.

class LlvmLibcMemccpyTest : public __llvm_libc::testing::Test {
public:
  void check_memccpy(__llvm_libc::cpp::MutableArrayRef<char> dst,
                     const __llvm_libc::cpp::ArrayRef<char> src, int end,
                     size_t count,
                     const __llvm_libc::cpp::ArrayRef<char> expected,
                     size_t expectedCopied, bool shouldReturnNull = false) {
    // Making sure we don't overflow buffer.
    ASSERT_GE(dst.size(), count);
    // Making sure memccpy returns dst.
    void *result = __llvm_libc::memccpy(dst.data(), src.data(), end, count);

    if (shouldReturnNull) {
      ASSERT_EQ(result, static_cast<void *>(nullptr));
    } else {
      ASSERT_EQ(result, static_cast<void *>(dst.data() + expectedCopied));
    }

    // Expected must be of the same size as dst.
    ASSERT_EQ(dst.size(), expected.size());
    // Expected and dst are the same.
    for (size_t i = 0; i < expected.size(); ++i)
      ASSERT_EQ(expected[i], dst[i]);
  }
};

TEST_F(LlvmLibcMemccpyTest, UntouchedUnrelatedEnd) {
  char dst[] = {'a', 'b'};
  const char src[] = {'x', '\0'};
  const char expected[] = {'a', 'b'};
  check_memccpy(dst, src, 'z', 0, expected, 0, true);
}

TEST_F(LlvmLibcMemccpyTest, UntouchedStartsWithEnd) {
  char dst[] = {'a', 'b'};
  const char src[] = {'x', '\0'};
  const char expected[] = {'a', 'b'};
  check_memccpy(dst, src, 'x', 0, expected, 0, true);
}

TEST_F(LlvmLibcMemccpyTest, CopyOneUnrelatedEnd) {
  char dst[] = {'a', 'b'};
  const char src[] = {'x', 'y'};
  const char expected[] = {'x', 'b'};
  check_memccpy(dst, src, 'z', 1, expected, 1, true);
}

TEST_F(LlvmLibcMemccpyTest, CopyOneStartsWithEnd) {
  char dst[] = {'a', 'b'};
  const char src[] = {'x', 'y'};
  const char expected[] = {'x', 'b'};
  check_memccpy(dst, src, 'x', 1, expected, 1);
}

TEST_F(LlvmLibcMemccpyTest, CopyTwoUnrelatedEnd) {
  char dst[] = {'a', 'b'};
  const char src[] = {'x', 'y'};
  const char expected[] = {'x', 'y'};
  check_memccpy(dst, src, 'z', 2, expected, 2, true);
}

TEST_F(LlvmLibcMemccpyTest, CopyTwoStartsWithEnd) {
  char dst[] = {'a', 'b'};
  const char src[] = {'x', 'y'};
  const char expected[] = {'x', 'b'};
  check_memccpy(dst, src, 'x', 2, expected, 1);
}
