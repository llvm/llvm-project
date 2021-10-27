//===-- Unittests for memmove ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/ArrayRef.h"
#include "src/string/memmove.h"
#include "utils/UnitTest/Test.h"

class LlvmLibcMemmoveTest : public __llvm_libc::testing::Test {
public:
  void check_memmove(void *dst, const void *src, size_t count,
                     const unsigned char *str,
                     const __llvm_libc::cpp::ArrayRef<unsigned char> expected) {
    void *result = __llvm_libc::memmove(dst, src, count);
    // Making sure the pointer returned is same with `dst`.
    EXPECT_EQ(result, dst);
    // `expected` is designed according to `str`.
    // `dst` and `src` might be part of `str`.
    // Making sure `str` is same with `expected`.
    for (size_t i = 0; i < expected.size(); ++i)
      EXPECT_EQ(str[i], expected[i]);
  }
};

TEST_F(LlvmLibcMemmoveTest, MoveZeroByte) {
  unsigned char dst[] = {'a', 'b'};
  const unsigned char src[] = {'y', 'z'};
  const unsigned char expected[] = {'a', 'b'};
  check_memmove(dst, src, 0, dst, expected);
}

TEST_F(LlvmLibcMemmoveTest, OverlapThatDstAndSrcPointToSameAddress) {
  unsigned char str[] = {'a', 'b'};
  const unsigned char expected[] = {'a', 'b'};
  check_memmove(str, str, 1, str, expected);
}

TEST_F(LlvmLibcMemmoveTest, OverlapThatDstStartsBeforeSrc) {
  // Set boundary at beginning and end for not overstepping when
  // copy forward or backward.
  unsigned char str[] = {'z', 'a', 'b', 'c', 'z'};
  const unsigned char expected[] = {'z', 'b', 'c', 'c', 'z'};
  // `dst` is `&str[1]`.
  check_memmove(&str[1], &str[2], 2, str, expected);
}

TEST_F(LlvmLibcMemmoveTest, OverlapThatDstStartsAfterSrc) {
  unsigned char str[] = {'z', 'a', 'b', 'c', 'z'};
  const unsigned char expected[] = {'z', 'a', 'a', 'b', 'z'};
  check_memmove(&str[2], &str[1], 2, str, expected);
}

// e.g. `dst` follow `src`.
// str: [abcdefghij]
//      [__src_____]
//      [_____dst__]
TEST_F(LlvmLibcMemmoveTest, SrcFollowDst) {
  unsigned char str[] = {'z', 'a', 'b', 'z'};
  const unsigned char expected[] = {'z', 'b', 'b', 'z'};
  check_memmove(&str[1], &str[2], 1, str, expected);
}

TEST_F(LlvmLibcMemmoveTest, DstFollowSrc) {
  unsigned char str[] = {'z', 'a', 'b', 'z'};
  const unsigned char expected[] = {'z', 'a', 'a', 'z'};
  check_memmove(&str[2], &str[1], 1, str, expected);
}
