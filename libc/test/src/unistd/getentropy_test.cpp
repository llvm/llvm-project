//===-- Unittests for getentropy ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/unistd/getentropy.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;

TEST(LlvmLibcUnistdGetEntropyTest, LengthTooLong) {
  char buf[1024];
  ASSERT_THAT(LIBC_NAMESPACE::getentropy(buf, 257), Fails(EIO));
  ASSERT_THAT(LIBC_NAMESPACE::getentropy(buf, 1024), Fails(EIO));
}

TEST(LlvmLibcUnistdGetEntropyTest, SmokeTest) {
  char buf[256];
  ASSERT_THAT(LIBC_NAMESPACE::getentropy(buf, 256), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::getentropy(buf, 0), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::getentropy(buf, 1), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::getentropy(buf, 16), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::getentropy(buf, 17), Succeeds());
}

TEST(LlvmLibcUnistdGetEntropyTest, OtherError) {
  ASSERT_THAT(LIBC_NAMESPACE::getentropy(nullptr, 1), Fails(EIO));
}
