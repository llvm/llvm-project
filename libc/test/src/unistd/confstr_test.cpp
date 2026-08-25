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

#include "hdr/errno_macros.h"
#include "hdr/types/size_t.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"

using LlvmLibcConfStrTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;

TEST_F(LlvmLibcConfStrTest, InvalidName) {
  char buf[64] = "initial";
  EXPECT_THAT(LIBC_NAMESPACE::confstr(0, buf, sizeof(buf)),
              Fails(EINVAL, size_t(0)));
  EXPECT_THAT(LIBC_NAMESPACE::confstr(0, nullptr, 0), Fails(EINVAL, size_t(0)));
  EXPECT_THAT(LIBC_NAMESPACE::confstr(-1, buf, sizeof(buf)),
              Fails(EINVAL, size_t(0)));
}
