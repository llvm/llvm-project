//===-- Unittests for sysconf ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/sysconf.h"
#include "test/UnitTest/Test.h"

#include <unistd.h>

TEST(LlvmLibcSysconfTest, PagesizeTest) {
  long pagesize = LIBC_NAMESPACE::sysconf(_SC_PAGESIZE);
  // TODO: fix page size support on RV32
  // (https://github.com/llvm/llvm-project/issues/162671)
#ifndef LIBC_TARGET_ARCH_IS_RISCV32
  ASSERT_GT(pagesize, 0l);
#endif
}
