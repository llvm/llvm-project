//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for sysinfo.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/types/struct_sysinfo.h"
#include "src/sys/sysinfo/sysinfo.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

TEST(LlvmLibcSysinfoTest, ValidBuffer) {
  struct sysinfo info;
  ASSERT_THAT(LIBC_NAMESPACE::sysinfo(&info), Succeeds(0));

  EXPECT_GT(info.uptime, 0l);
  EXPECT_GT(info.totalram, 0ul);
  EXPECT_GT(info.mem_unit, 0u);
  EXPECT_GT(info.procs, static_cast<unsigned short>(0));
}

TEST(LlvmLibcSysinfoTest, NullptrBuffer) {
  EXPECT_THAT(LIBC_NAMESPACE::sysinfo(nullptr), Fails(EFAULT));
}
