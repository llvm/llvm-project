//===-- Unittests for uname -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/string_view.h"
#include "src/__support/macros/properties/architectures.h"
#include "src/sys/utsname/uname.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <errno.h>
#include <sys/utsname.h>

TEST(LlvmLibcUnameTest, GetMachineName) {
  struct utsname names;
  ASSERT_GE(__llvm_libc::uname(&names), 0);
#ifdef LIBC_TARGET_ARCH_IS_X86_64
  ASSERT_STREQ(names.machine, "x86_64");
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64)
  ASSERT_STREQ(names.machine, "aarch64");
#endif
}
