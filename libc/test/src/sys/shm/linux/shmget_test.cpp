//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for shmget.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/sys_ipc_macros.h"
#include "hdr/sys_stat_macros.h"
#include "src/__support/CPP/scope.h"
#include "src/sys/shm/shmctl.h"
#include "src/sys/shm/shmget.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LIBC_NAMESPACE::cpp::scope_exit;
using LlvmLibcShmGetTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcShmGetTest, AllocatesSegment) {
  int shmid;
  ASSERT_THAT(shmid = LIBC_NAMESPACE::shmget(
                  IPC_PRIVATE, 1, IPC_CREAT | IPC_EXCL | S_IRUSR | S_IWUSR),
              returns(GE(0)).with_errno(EQ(0)));
  scope_exit cleanup([&] {
    EXPECT_THAT(LIBC_NAMESPACE::shmctl(shmid, IPC_RMID, nullptr), Succeeds(0));
  });
}

TEST_F(LlvmLibcShmGetTest, InvalidSizeReturnsEinval) {
  // Creating a new segment with size 0 (< SHMMIN) always fails with EINVAL.
  EXPECT_THAT(
      LIBC_NAMESPACE::shmget(IPC_PRIVATE, 0, IPC_CREAT | S_IRUSR | S_IWUSR),
      Fails(EINVAL));
}
