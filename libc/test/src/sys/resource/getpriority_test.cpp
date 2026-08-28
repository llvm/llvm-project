
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for getpriority.
///
//===----------------------------------------------------------------------===//

#include "hdr/sys_resource_macros.h"
#include "hdr/types/id_t.h"
#include "src/sys/resource/getpriority.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcGetrusageTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcGetrusageTest, BasicTest) {
  int nice = LIBC_NAMESPACE::getpriority(PRIO_PROCESS, 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GE(nice, -20);
  ASSERT_LE(nice, 19);

}

