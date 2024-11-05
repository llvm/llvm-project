//===-- Unittests for process_mrelease ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/mman/process_mrelease.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/LibcTest.h"

#include <sys/syscall.h>

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;

#if defined(SYS_process_mrelease)
TEST(LlvmLibcProcessMReleaseTest, ErrorNonExistingPidfd) {
  EXPECT_THAT(LIBC_NAMESPACE::process_mrelease(-1, 0), Fails(EBADF));
}
#endif
