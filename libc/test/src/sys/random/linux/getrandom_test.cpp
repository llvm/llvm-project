//===-- Unittests for getrandom -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/math/fabs.h"
#include "src/sys/random/getrandom.h"
#include "test/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcGetRandomTest, InvalidFlag) {
  using __llvm_libc::testing::ErrnoSetterMatcher::Fails;
  static constexpr size_t SIZE = 256;
  char data[SIZE];
  libc_errno = 0;
  ASSERT_THAT(__llvm_libc::getrandom(data, SIZE, -1), Fails(EINVAL));
  libc_errno = 0;
}

TEST(LlvmLibcGetRandomTest, InvalidBuffer) {
  using __llvm_libc::testing::ErrnoSetterMatcher::Fails;

  libc_errno = 0;
  ASSERT_THAT(__llvm_libc::getrandom(nullptr, 65536, 0), Fails(EFAULT));
  libc_errno = 0;
}

TEST(LlvmLibcGetRandomTest, ReturnsSize) {
  static constexpr size_t SIZE = 8192;
  uint8_t buf[SIZE];
  for (size_t i = 0; i < SIZE; ++i) {
    // Without GRND_RANDOM set this should never fail.
    ASSERT_EQ(__llvm_libc::getrandom(buf, i, 0), static_cast<ssize_t>(i));
  }
}

TEST(LlvmLibcGetRandomTest, PiEstimation) {
  static constexpr size_t LIMIT = 10000000;
  static constexpr double PI = 3.14159265358979;

  auto generator = []() {
    uint16_t data;
    __llvm_libc::getrandom(&data, sizeof(data), 0);
    return data;
  };

  auto sample = [&]() {
    auto x = static_cast<double>(generator()) / 65536.0;
    auto y = static_cast<double>(generator()) / 65536.0;
    return x * x + y * y < 1.0;
  };

  double counter = 0;
  for (size_t i = 0; i < LIMIT; ++i) {
    if (sample()) {
      counter += 1.0;
    }
  }
  counter = counter / LIMIT * 4.0;
  ASSERT_TRUE(__llvm_libc::fabs(counter - PI) < 0.1);
}
