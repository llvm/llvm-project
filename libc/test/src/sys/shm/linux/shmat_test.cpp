//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for shmat.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/stdint_proxy.h"
#include "hdr/sys_ipc_macros.h"
#include "hdr/sys_shm_macros.h"
#include "hdr/sys_stat_macros.h"
#include "src/__support/CPP/scope.h"
#include "src/sys/shm/shmat.h"
#include "src/sys/shm/shmctl.h"
#include "src/sys/shm/shmdt.h"
#include "src/sys/shm/shmget.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LIBC_NAMESPACE::cpp::scope_exit;
using LlvmLibcShmAtTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcShmAtTest, AttachReadWriteAndReadOnly) {
  int shmid;
  ASSERT_THAT(shmid = LIBC_NAMESPACE::shmget(
                  IPC_PRIVATE, 256, IPC_CREAT | IPC_EXCL | S_IRUSR | S_IWUSR),
              returns(GE(0)).with_errno(EQ(0)));
  scope_exit cleanup([&] {
    EXPECT_THAT(LIBC_NAMESPACE::shmctl(shmid, IPC_RMID, nullptr), Succeeds(0));
  });

  // Attach the newly created segment for read-write access.
  void *rw_addr = LIBC_NAMESPACE::shmat(shmid, nullptr, 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_NE(rw_addr, SHM_FAILED);

  // Newly created System V shared memory segments are zero-initialized.
  int *rw_val = static_cast<int *>(rw_addr);
  EXPECT_EQ(*rw_val, 0);
  *rw_val = 42;

  // Attach the same segment a second time with SHM_RDONLY and verify that
  // the modification is visible through the second mapping.
  void *ro_addr = LIBC_NAMESPACE::shmat(shmid, nullptr, SHM_RDONLY);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_NE(ro_addr, SHM_FAILED);
  EXPECT_NE(ro_addr, rw_addr);
  EXPECT_EQ(*static_cast<const int *>(ro_addr), 42);

  EXPECT_THAT(LIBC_NAMESPACE::shmdt(rw_addr), Succeeds(0));
  EXPECT_THAT(LIBC_NAMESPACE::shmdt(ro_addr), Succeeds(0));
}

TEST_F(LlvmLibcShmAtTest, UnalignedAddrWithoutShmRndFails) {
  int shmid;
  ASSERT_THAT(shmid = LIBC_NAMESPACE::shmget(
                  IPC_PRIVATE, 256, IPC_CREAT | IPC_EXCL | S_IRUSR | S_IWUSR),
              returns(GE(0)).with_errno(EQ(0)));
  scope_exit cleanup([&] {
    EXPECT_THAT(LIBC_NAMESPACE::shmctl(shmid, IPC_RMID, nullptr), Succeeds(0));
  });

  void *addr = LIBC_NAMESPACE::shmat(shmid, nullptr, 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_NE(addr, SHM_FAILED);

  // Unaligned shmaddr without SHM_RND must fail with EINVAL (even with
  // SHM_REMAP allowing replacement of the existing mapping at addr).
  EXPECT_THAT(LIBC_NAMESPACE::shmat(shmid, static_cast<const char *>(addr) + 1,
                                    SHM_REMAP),
              Fails(EINVAL, SHM_FAILED));

  EXPECT_THAT(LIBC_NAMESPACE::shmdt(addr), Succeeds(0));
}

TEST_F(LlvmLibcShmAtTest, UnalignedAddrWithShmRndRoundsDown) {
  int shmid;
  ASSERT_THAT(shmid = LIBC_NAMESPACE::shmget(
                  IPC_PRIVATE, 256, IPC_CREAT | IPC_EXCL | S_IRUSR | S_IWUSR),
              returns(GE(0)).with_errno(EQ(0)));
  scope_exit cleanup([&] {
    EXPECT_THAT(LIBC_NAMESPACE::shmctl(shmid, IPC_RMID, nullptr), Succeeds(0));
  });

  void *addr = LIBC_NAMESPACE::shmat(shmid, nullptr, 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_NE(addr, SHM_FAILED);
  ASSERT_THAT(LIBC_NAMESPACE::shmdt(addr), Succeeds(0));

  // With SHM_RND, the unaligned address (addr + 1) is rounded down to the
  // nearest multiple of SHMLBA (addr).
  void *rounded_addr = LIBC_NAMESPACE::shmat(
      shmid, static_cast<const char *>(addr) + 1, SHM_RND);
  ASSERT_ERRNO_SUCCESS();
  EXPECT_EQ(reinterpret_cast<uintptr_t>(rounded_addr) % SHMLBA, uintptr_t(0));
  EXPECT_EQ(rounded_addr, addr);

  EXPECT_THAT(LIBC_NAMESPACE::shmdt(rounded_addr), Succeeds(0));
}

TEST_F(LlvmLibcShmAtTest, InvalidShmidReturnsEinval) {
  EXPECT_THAT(LIBC_NAMESPACE::shmat(-1, nullptr, 0), Fails(EINVAL, SHM_FAILED));
}
