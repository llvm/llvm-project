//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for shmdt.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/sys_ipc_macros.h"
#include "hdr/sys_shm_macros.h"
#include "hdr/sys_stat_macros.h"
#include "src/sys/shm/shmat.h"
#include "src/sys/shm/shmdt.h"
#include "src/sys/shm/shmget.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcShmDtTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcShmDtTest, DetachSegment) {
  int shmid;
  ASSERT_THAT(shmid = LIBC_NAMESPACE::shmget(
                  IPC_PRIVATE, 256, IPC_CREAT | IPC_EXCL | S_IRUSR | S_IWUSR),
              returns(GE(0)).with_errno(EQ(0)));

  void *addr = LIBC_NAMESPACE::shmat(shmid, nullptr, 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_NE(addr, SHM_FAILED);

  // Detaching at an offset inside the attached segment must fail with EINVAL.
  EXPECT_THAT(LIBC_NAMESPACE::shmdt(static_cast<const char *>(addr) + 1),
              Fails(EINVAL));

  // Detaching at the start address of the attached segment succeeds.
  ASSERT_THAT(LIBC_NAMESPACE::shmdt(addr), Succeeds(0));

  // Detaching an already detached address must fail with EINVAL.
  EXPECT_THAT(LIBC_NAMESPACE::shmdt(addr), Fails(EINVAL));

  // TODO: Remove the segment with shmctl once shmctl is implemented.
}

TEST_F(LlvmLibcShmDtTest, InvalidAddrReturnsEinval) {
  int stack_var = 0;
  EXPECT_THAT(LIBC_NAMESPACE::shmdt(nullptr), Fails(EINVAL));
  EXPECT_THAT(LIBC_NAMESPACE::shmdt(&stack_var), Fails(EINVAL));
}
