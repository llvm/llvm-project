//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for confstr
///
//===----------------------------------------------------------------------===//

#include "src/unistd/confstr.h"

#include "hdr/types/size_t.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcConfStrTest, Basic) {
  char buf[64] = "initial";
  size_t ret = LIBC_NAMESPACE::confstr(0, buf, sizeof(buf));
  EXPECT_EQ(ret, size_t(0));
}

TEST(LlvmLibcConfStrTest, NullBufZeroLen) {
  size_t ret = LIBC_NAMESPACE::confstr(0, nullptr, 0);
  EXPECT_EQ(ret, size_t(0));
}

TEST(LlvmLibcConfStrTest, NonExistentConfig) {
  char buf[64];
  size_t ret = LIBC_NAMESPACE::confstr(-1, buf, sizeof(buf));
  EXPECT_EQ(ret, size_t(0));
}
