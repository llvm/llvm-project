/* Check if clone (CLONE_THREAD) does not call exit_group (BZ #21512)
   Copyright (C) 2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <string.h>
#include <sched.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <linux/futex.h>
#include <support/check.h>
#include <stdatomic.h>
#include <clone_internal.h>

/* Test if clone call with CLONE_THREAD does not call exit_group.  The 'f'
   function returns '1', which will be used by clone thread to call the
   'exit' syscall directly.  If _exit is used instead, exit_group will be
   used and thus the thread group will finish with return value of '1'
   (where '2' from main thread is expected.).  */

static int
f (void *a)
{
  return 1;
}

/* Futex wait for TID argument, similar to pthread_join internal
   implementation.  */
#define wait_tid(ctid_ptr, ctid_val)					\
  do {									\
    __typeof (*(ctid_ptr)) __tid;					\
    /* We need acquire MO here so that we synchronize with the		\
       kernel's store to 0 when the clone terminates.  */		\
    while ((__tid = atomic_load_explicit (ctid_ptr,			\
					  memory_order_acquire)) != 0)	\
      futex_wait (ctid_ptr, ctid_val);					\
  } while (0)

static inline int
futex_wait (int *futexp, int val)
{
#ifdef __NR_futex
  return syscall (__NR_futex, futexp, FUTEX_WAIT, val);
#else
  return syscall (__NR_futex_time64, futexp, FUTEX_WAIT, val);
#endif
}

static int
do_test (void)
{
  char st[1024] __attribute__ ((aligned));
  int clone_flags = CLONE_THREAD;
  /* Minimum required flags to used along with CLONE_THREAD.  */
  clone_flags |= CLONE_VM | CLONE_SIGHAND;
  /* We will used ctid to call on futex to wait for thread exit.  */
  clone_flags |= CLONE_CHILD_CLEARTID;
  /* Initialize with a known value.  ctid is set to zero by the kernel after the
     cloned thread has exited.  */
#define CTID_INIT_VAL 1
  pid_t ctid = CTID_INIT_VAL;
  pid_t tid;

  struct clone_args clone_args =
    {
      .flags = clone_flags & ~CSIGNAL,
      .exit_signal = clone_flags & CSIGNAL,
      .stack = (uintptr_t) st,
      .stack_size = sizeof (st),
      .child_tid = (uintptr_t) &ctid,
    };
  tid = __clone_internal (&clone_args, f, NULL);
  if (tid == -1)
    FAIL_EXIT1 ("clone failed: %m");

  wait_tid (&ctid, CTID_INIT_VAL);

  return 2;
}

#define EXPECTED_STATUS 2
#include <support/test-driver.c>
