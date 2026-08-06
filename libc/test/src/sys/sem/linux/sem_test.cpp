//===-- Unittests for sys/sem.h entrypoints ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/sys_ipc_macros.h"
#include "hdr/sys_sem_macros.h"
#include "hdr/types/struct_sembuf.h"
#include "hdr/types/struct_semid_ds.h"
#include "src/sys/sem/semctl.h"
#include "src/sys/sem/semget.h"
#include "src/sys/sem/semop.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
using LlvmLibcSysSemTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

union semun {
  int val;
  struct semid_ds *buf;
  unsigned short *array;
};

TEST_F(LlvmLibcSysSemTest, SemgetSemctlSemopFlow) {

  // create a semaphore
  int semid =
      LIBC_NAMESPACE::semget(IPC_PRIVATE, 1, IPC_CREAT | IPC_EXCL | 0600);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(semid, -1);

  union semun set_val;
  set_val.val = 1;

  // set the semaphore value to 1
  ASSERT_THAT(LIBC_NAMESPACE::semctl(semid, 0, SETVAL, set_val), Succeeds(0));

  // get the value of semaphore should be 1
  ASSERT_THAT(LIBC_NAMESPACE::semctl(semid, 0, GETVAL), Succeeds(1));

  // decrement the semaphore value with IPC_NOWAIT flag
  struct sembuf decrement_op = {0, -1, IPC_NOWAIT};
  ASSERT_THAT(LIBC_NAMESPACE::semop(semid, &decrement_op, 1), Succeeds(0));

  // get the value of semaphore should be 0
  ASSERT_THAT(LIBC_NAMESPACE::semctl(semid, 0, GETVAL), Succeeds(0));

  // increment the semaphore with IPC_NOWAIT flag
  struct sembuf increment_op = {0, 1, IPC_NOWAIT};
  ASSERT_THAT(LIBC_NAMESPACE::semop(semid, &increment_op, 1), Succeeds(0));

  // get the semaphore value should be 1
  ASSERT_THAT(LIBC_NAMESPACE::semctl(semid, 0, GETVAL), Succeeds(1));

  // get the IPC stats
  struct semid_ds sem_ds;
  union semun stat_arg;
  stat_arg.buf = &sem_ds;
  ASSERT_THAT(LIBC_NAMESPACE::semctl(semid, 0, IPC_STAT, stat_arg),
              Succeeds(0));

  // the number of sem is 1
  ASSERT_EQ(sem_ds.sem_nsems, 1UL);

  // destroy the semaphore
  ASSERT_THAT(LIBC_NAMESPACE::semctl(semid, 0, IPC_RMID), Succeeds(0));
}
