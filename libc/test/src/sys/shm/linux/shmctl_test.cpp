//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for shmctl.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/sys_ipc_macros.h"
#include "hdr/sys_stat_macros.h"
#include "hdr/types/struct_shmid_ds.h"
#include "src/__support/CPP/scope.h"
#include "src/sys/shm/shmctl.h"
#include "src/sys/shm/shmget.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LIBC_NAMESPACE::cpp::scope_exit;
using LlvmLibcShmCtlTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcShmCtlTest, StatSetAndRemove) {
  int shmid;
  ASSERT_THAT(shmid = LIBC_NAMESPACE::shmget(
                  IPC_PRIVATE, 256, IPC_CREAT | IPC_EXCL | S_IRUSR | S_IWUSR),
              returns(GE(0)).with_errno(EQ(0)));
  scope_exit cleanup([&] {
    EXPECT_THAT(LIBC_NAMESPACE::shmctl(shmid, IPC_RMID, nullptr), Succeeds(0));
  });

  struct shmid_ds ds = {};
  ASSERT_THAT(LIBC_NAMESPACE::shmctl(shmid, IPC_STAT, &ds), Succeeds(0));
  EXPECT_EQ(ds.shm_segsz, size_t(256));
  EXPECT_EQ(ds.shm_nattch, shmatt_t(0));
  EXPECT_EQ(ds.shm_perm.mode & 0777, mode_t(S_IRUSR | S_IWUSR));

  ds.shm_perm.mode = S_IRUSR;
  ASSERT_THAT(LIBC_NAMESPACE::shmctl(shmid, IPC_SET, &ds), Succeeds(0));

  struct shmid_ds updated_ds = {};
  ASSERT_THAT(LIBC_NAMESPACE::shmctl(shmid, IPC_STAT, &updated_ds),
              Succeeds(0));
  EXPECT_EQ(updated_ds.shm_perm.mode & 0777, mode_t(S_IRUSR));

  ASSERT_THAT(LIBC_NAMESPACE::shmctl(shmid, IPC_RMID, nullptr), Succeeds(0));
  cleanup.release();

  // Verify that once shmid is deleted, we can no longer query stat.
  EXPECT_THAT(LIBC_NAMESPACE::shmctl(shmid, IPC_STAT, &ds), Fails(EINVAL));
}

TEST_F(LlvmLibcShmCtlTest, InvalidArgsReturnEinval) {
  struct shmid_ds ds = {};
  EXPECT_THAT(LIBC_NAMESPACE::shmctl(-1, IPC_STAT, &ds), Fails(EINVAL));
  EXPECT_THAT(LIBC_NAMESPACE::shmctl(-1, -1, &ds), Fails(EINVAL));
}
