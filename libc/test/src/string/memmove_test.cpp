//===-- Unittests for memmove ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/memcmp.h"
#include "src/string/memmove.h"
#include "utils/CPP/ArrayRef.h"
#include "utils/UnitTest/Test.h"

class LlvmLibcMemmoveTest : public __llvm_libc::testing::Test {
public:
  void check_memmove(void *dest, const void *src, size_t count, const void *str,
                     const __llvm_libc::cpp::ArrayRef<unsigned char> expected) {
    void *result = __llvm_libc::memmove(dest, src, count);
    // Making sure the pointer returned is same with dest.
    EXPECT_EQ(result, dest);
    // expected is designed according to str.
    // dest and src might be part of str.
    // Making sure the str is same with expected.
    EXPECT_EQ(__llvm_libc::memcmp(str, expected.data(), expected.size()), 0);
  }
};

TEST_F(LlvmLibcMemmoveTest, MoveZeroByte) {
  unsigned char dest[] = {'a', 'b'};
  const unsigned char src[] = {'y', 'z'};
  const unsigned char expected[] = {'a', 'b'};
  check_memmove(dest, src, 0, dest, expected);
}

TEST_F(LlvmLibcMemmoveTest, OverlapThatDestAndSrcPointToSameAddress) {
  unsigned char str[] = {'a', 'b'};
  const unsigned char expected[] = {'a', 'b'};
  check_memmove(str, str, 1, str, expected);
}

TEST_F(LlvmLibcMemmoveTest, OverlapThatDestStartsBeforeSrc) {
  // Set boundary at beginning and end for not overstepping when
  // copy forward or backward.
  unsigned char str[] = {'z', 'a', 'b', 'c', 'z'};
  const unsigned char expected[] = {'z', 'b', 'c', 'c', 'z'};
  // dest is &str[1].
  check_memmove(&str[1], &str[2], 2, str, expected);
}

TEST_F(LlvmLibcMemmoveTest, OverlapThatDestStartsAfterSrc) {
  unsigned char str[] = {'z', 'a', 'b', 'c', 'z'};
  const unsigned char expected[] = {'z', 'a', 'a', 'b', 'z'};
  check_memmove(&str[2], &str[1], 2, str, expected);
}

// e.g. dest follow src.
// str: [abcdefghij]
//      [__src_____]
//      [_____dest_]
TEST_F(LlvmLibcMemmoveTest, SrcFollowDest) {
  unsigned char str[] = {'z', 'a', 'b', 'z'};
  const unsigned char expected[] = {'z', 'b', 'b', 'z'};
  check_memmove(&str[1], &str[2], 1, str, expected);
}

TEST_F(LlvmLibcMemmoveTest, DestFollowSrc) {
  unsigned char str[] = {'z', 'a', 'b', 'z'};
  const unsigned char expected[] = {'z', 'a', 'a', 'z'};
  check_memmove(&str[2], &str[1], 1, str, expected);
}
