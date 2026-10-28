//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for the getentropy function.
///
//===----------------------------------------------------------------------===//

#include "src/sys/random/getentropy.h"

#include "hdr/errno_macros.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcGetEntropyTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcGetEntropyTest, ZeroLength) {
  char buffer[16]{};
  ASSERT_THAT(LIBC_NAMESPACE::getentropy(buffer, 0), Succeeds<int>(0));
  ASSERT_THAT(LIBC_NAMESPACE::getentropy(nullptr, 0), Succeeds<int>(0));
}

TEST_F(LlvmLibcGetEntropyTest, LengthTooLarge) {
  char buffer[257]{};
  ASSERT_THAT(LIBC_NAMESPACE::getentropy(buffer, 257), Fails<int>(EIO));
}

TEST_F(LlvmLibcGetEntropyTest, InvalidBuffer) {
  ASSERT_THAT(LIBC_NAMESPACE::getentropy(nullptr, 16), Fails<int>(EFAULT));
}

TEST_F(LlvmLibcGetEntropyTest, SmallBuffer) {
  char buffer[16]{};
  ASSERT_THAT(LIBC_NAMESPACE::getentropy(buffer, sizeof(buffer)),
              Succeeds<int>(0));
}

TEST_F(LlvmLibcGetEntropyTest, MaxBuffer) {
  constexpr size_t MAX_BYTES = 256;
  char buffer[MAX_BYTES]{};
  ASSERT_THAT(LIBC_NAMESPACE::getentropy(buffer, MAX_BYTES), Succeeds<int>(0));

  bool all_zeros = true;
  for (size_t i = 0; i < MAX_BYTES; ++i) {
    if (buffer[i] != 0) {
      all_zeros = false;
      break;
    }
  }
  ASSERT_FALSE(all_zeros);
}

TEST_F(LlvmLibcGetEntropyTest, DifferentOutputs) {
  constexpr size_t BUF_SIZE = 32;
  char buf1[BUF_SIZE]{};
  char buf2[BUF_SIZE]{};
  ASSERT_THAT(LIBC_NAMESPACE::getentropy(buf1, BUF_SIZE), Succeeds<int>(0));
  ASSERT_THAT(LIBC_NAMESPACE::getentropy(buf2, BUF_SIZE), Succeeds<int>(0));

  bool differ = false;
  for (size_t i = 0; i < BUF_SIZE; ++i) {
    if (buf1[i] != buf2[i]) {
      differ = true;
      break;
    }
  }
  ASSERT_TRUE(differ);
}
