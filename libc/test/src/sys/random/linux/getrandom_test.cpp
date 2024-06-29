//===-- Unittests for getrandom -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/array.h"
#include "src/errno/libc_errno.h"
#include "src/math/fabs.h"
#include "src/sys/random/getrandom.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcGetRandomTest, InvalidFlag) {
  LIBC_NAMESPACE::cpp::array<char, 10> buffer;
  LIBC_NAMESPACE::libc_errno = 0;
  ASSERT_THAT(LIBC_NAMESPACE::getrandom(buffer.data(), buffer.size(), -1),
              LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails(EINVAL));
}

TEST(LlvmLibcGetRandomTest, InvalidBuffer) {
  LIBC_NAMESPACE::libc_errno = 0;
  ASSERT_THAT(LIBC_NAMESPACE::getrandom(nullptr, 65536, 0),
              LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails(EFAULT));
}

TEST(LlvmLibcGetRandomTest, ReturnsSize) {
  LIBC_NAMESPACE::cpp::array<char, 10> buffer;
  for (size_t i = 0; i < buffer.size(); ++i) {
    // Without GRND_RANDOM set this should never fail.
    ASSERT_EQ(LIBC_NAMESPACE::getrandom(buffer.data(), i, 0),
              static_cast<ssize_t>(i));
  }
}

TEST(LlvmLibcGetRandomTest, CheckValue) {
  // Probability of picking one particular value amongst 256 possibilities a
  // hundred times in a row is (1/256)^100 = 1.49969681e-241.
  LIBC_NAMESPACE::cpp::array<char, 100> buffer;

  for (char &c : buffer)
    c = 0;

  LIBC_NAMESPACE::getrandom(buffer.data(), buffer.size(), 0);

  bool all_zeros = true;
  for (char c : buffer)
    if (c != 0)
      all_zeros = false;

  ASSERT_FALSE(all_zeros);
}
