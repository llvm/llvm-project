//===-- Failure unittests for select --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/select/select.h"
#include "src/unistd/read.h"
#include "test/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <errno.h>
#include <sys/select.h>
#include <unistd.h>

using __llvm_libc::testing::ErrnoSetterMatcher::Fails;

TEST(LlvmLibcSelectTest, SelectInvalidFD) {
  fd_set set;
  FD_ZERO(&set);
  struct timeval timeout {
    0, 0
  };
  ASSERT_THAT(__llvm_libc::select(-1, &set, nullptr, nullptr, &timeout),
              Fails(EINVAL));
}
