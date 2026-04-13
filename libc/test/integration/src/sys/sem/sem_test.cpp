//===-- Integration test for sys/sem --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/threads/sleep.h"
#include "src/fcntl/open.h"
#include "src/signal/kill.h"
#include "src/stdlib/exit.h"
#include "src/sys/ipc/ftok.h"
#include "src/sys/sem/semctl.h"
#include "src/sys/sem/semget.h"
#include "src/sys/sem/semop.h"
#include "src/sys/wait/waitpid.h"
#include "src/unistd/close.h"
#include "src/unistd/fork.h"
#include "src/unistd/unlink.h"

#include "test/IntegrationTest/test.h"

#include "hdr/fcntl_macros.h"
#include "hdr/sys_ipc_macros.h"
#include "hdr/sys_sem_macros.h"
#include "hdr/types/struct_sembuf.h"

#include <errno.h>
#include <signal.h>
#include <sys/wait.h>

// child try IPC_CREAT|IPC_EXCL,
// but expect EEXIST then fall back to get the existing semaphore.
static int sem_joiner_get(key_t semkey) {
  // this expect to fail and return -1 because we used the same key
  // and this semaphore has been initialized.
  int sid = LIBC_NAMESPACE::semget(semkey, 1, IPC_CREAT | IPC_EXCL | 0666);
  if (sid == -1 && errno == EEXIST)
    // get the initialized semaphore id
    sid = LIBC_NAMESPACE::semget(semkey, 0, 0);
  return sid;
}

// fork a child that acquires the semaphore and hold untill killed.
// returns child pid.
static pid_t fork_sem_holder(key_t semkey) {
  pid_t pid = LIBC_NAMESPACE::fork();
  if (pid == 0) {
    int sid = sem_joiner_get(semkey);
    if (sid < 0)
      LIBC_NAMESPACE::exit(1);
    // try acquire and hold
    // set flag to UNDO if process terminate unexpectedly
    struct sembuf acq = {0, -1, SEM_UNDO};
    if (LIBC_NAMESPACE::semop(sid, &acq, 1) != 0)
      LIBC_NAMESPACE::exit(2);

    // hold the semaphore until killed by parent.
    while (1)
      LIBC_NAMESPACE::sleep_briefly();
  }
  return pid;
}

// fork a child that tries to acquire (IPC_NOWAIT).
// exit code: 0 = acquired, 1 = EAGAIN (full, try later again), 2+ = error.
static pid_t fork_sem_try_acquire(key_t semkey) {
  pid_t pid = LIBC_NAMESPACE::fork();
  if (pid == 0) {
    int sid = sem_joiner_get(semkey);
    if (sid < 0)
      LIBC_NAMESPACE::exit(3);
    // try acquire
    struct sembuf acq = {0, -1, SEM_UNDO | IPC_NOWAIT};

    if (LIBC_NAMESPACE::semop(sid, &acq, 1) == 0)
      LIBC_NAMESPACE::exit(0); // acquired

    // exit with code 1 if EAGAIN
    LIBC_NAMESPACE::exit(errno == EAGAIN ? 1 : 2);
  }
  return pid;
}

// wait for child to terminate.
static int wait_for_child(pid_t pid) {
  int status;
  LIBC_NAMESPACE::waitpid(pid, &status, 0);
  return status;
}

// semaphore gate that allows at most two processes to enter
void sem_gate_test() {
  constexpr const char *TEST_FILE = "/tmp/sem_gate_test";
  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_CREAT | O_WRONLY, 0666);
  ASSERT_TRUE(fd >= 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_TRUE(LIBC_NAMESPACE::close(fd) == 0);

  key_t semkey = LIBC_NAMESPACE::ftok(TEST_FILE, 'a');
  ASSERT_TRUE(semkey != key_t(-1));
  ASSERT_ERRNO_SUCCESS();

  // clean up stale semaphore from previous run because semaphore
  // are kernel persistent, they can remain on the system if previous
  // test fail and exit.
  int old_semid = LIBC_NAMESPACE::semget(semkey, 1, 0);
  if (old_semid != -1)
    LIBC_NAMESPACE::semctl(old_semid, 0, IPC_RMID);
  errno = 0;

  // create a semaphore
  int semid = LIBC_NAMESPACE::semget(semkey, 1, IPC_CREAT | IPC_EXCL | 0666);
  ASSERT_TRUE(semid >= 0);
  ASSERT_ERRNO_SUCCESS();

  // increment the first semaphore by 2 that allows two processes to access
  struct sembuf sbuf = {0, 2, 0};
  ASSERT_TRUE(LIBC_NAMESPACE::semop(semid, &sbuf, 1) == 0);

  // verify the first semaphore value
  ASSERT_TRUE(LIBC_NAMESPACE::semctl(semid, 0, GETVAL) == 2);

  // fork two children to hold the semaphore
  pid_t holder1 = fork_sem_holder(semkey);
  ASSERT_TRUE(holder1 > 0);
  pid_t holder2 = fork_sem_holder(semkey);
  ASSERT_TRUE(holder2 > 0);

  // wait until both children have acquired so semaphore value reaches 0.
  while (LIBC_NAMESPACE::semctl(semid, 0, GETVAL) > 0)
    LIBC_NAMESPACE::sleep_briefly();

  // check my current semaphore value is 0
  ASSERT_TRUE(LIBC_NAMESPACE::semctl(semid, 0, GETVAL) == 0);

  // now the semaphore gate is full
  // fork another process to try to acquire it
  pid_t blocked = fork_sem_try_acquire(semkey);
  ASSERT_TRUE(blocked > 0);
  int blocked_status = wait_for_child(blocked);
  // check the exit status
  ASSERT_TRUE(WIFEXITED(blocked_status));
  ASSERT_EQ(WEXITSTATUS(blocked_status), 1); // EAGAIN (blocked)

  // kill the first holder to make one slot avalaible in semaphore
  LIBC_NAMESPACE::kill(holder1, SIGKILL);
  wait_for_child(holder1);
  ASSERT_TRUE(LIBC_NAMESPACE::semctl(semid, 0, GETVAL) == 1);

  // then fork child again to try acquire
  pid_t unblocked = fork_sem_try_acquire(semkey);
  ASSERT_TRUE(unblocked > 0);
  int unblocked_status = wait_for_child(unblocked);
  // this child should get it since the slot is avalaible now
  ASSERT_TRUE(WIFEXITED(unblocked_status));
  ASSERT_EQ(WEXITSTATUS(unblocked_status), 0); // acquired

  // cleanup
  LIBC_NAMESPACE::kill(holder2, SIGKILL);
  wait_for_child(holder2);
  ASSERT_TRUE(LIBC_NAMESPACE::semctl(semid, 0, IPC_RMID) == 0);
  ASSERT_TRUE(LIBC_NAMESPACE::unlink(TEST_FILE) == 0);
}

TEST_MAIN([[maybe_unused]] int argc, [[maybe_unused]] char **argv,
          [[maybe_unused]] char **envp) {
  sem_gate_test();
  return 0;
}
