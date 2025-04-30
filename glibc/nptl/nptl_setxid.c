/* Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2002.

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

#include <futex-internal.h>
#include <ldsodefs.h>
#include <list.h>
#include <lowlevellock.h>
#include <pthreadP.h>
#include <unistd.h>

/* Check for consistency across set*id system call results.  The abort
   should not happen as long as all privileges changes happen through
   the glibc wrappers.  ERROR must be 0 (no error) or an errno
   code.  */
static void
setxid_error (struct xid_command *cmdp, int error)
{
  do
    {
      int olderror = cmdp->error;
      if (olderror == error)
        break;
      if (olderror != -1)
        {
          /* Mismatch between current and previous results.  Save the
             error value to memory so that is not clobbered by the
             abort function and preserved in coredumps.  */
          volatile int xid_err __attribute__ ((unused)) = error;
          abort ();
        }
    }
  while (atomic_compare_and_exchange_bool_acq (&cmdp->error, error, -1));
}

/* Set by __nptl_setxid and used by __nptl_setxid_sighandler.  */
static struct xid_command *xidcmd;

/* We use the SIGSETXID signal in the setuid, setgid, etc. implementations to
   tell each thread to call the respective setxid syscall on itself.  This is
   the handler.  */
void
__nptl_setxid_sighandler (int sig, siginfo_t *si, void *ctx)
{
  int result;

  /* Safety check.  It would be possible to call this function for
     other signals and send a signal from another process.  This is not
     correct and might even be a security problem.  Try to catch as
     many incorrect invocations as possible.  */
  if (sig != SIGSETXID
      || si->si_pid != __getpid ()
      || si->si_code != SI_TKILL)
    return;

  result = INTERNAL_SYSCALL_NCS (xidcmd->syscall_no, 3, xidcmd->id[0],
				 xidcmd->id[1], xidcmd->id[2]);
  int error = 0;
  if (__glibc_unlikely (INTERNAL_SYSCALL_ERROR_P (result)))
    error = INTERNAL_SYSCALL_ERRNO (result);
  setxid_error (xidcmd, error);

  /* Reset the SETXID flag.  */
  struct pthread *self = THREAD_SELF;
  int flags, newval;
  do
    {
      flags = THREAD_GETMEM (self, cancelhandling);
      newval = THREAD_ATOMIC_CMPXCHG_VAL (self, cancelhandling,
					  flags & ~SETXID_BITMASK, flags);
    }
  while (flags != newval);

  /* And release the futex.  */
  self->setxid_futex = 1;
  futex_wake (&self->setxid_futex, 1, FUTEX_PRIVATE);

  if (atomic_decrement_val (&xidcmd->cntr) == 0)
    futex_wake ((unsigned int *) &xidcmd->cntr, 1, FUTEX_PRIVATE);
}
libc_hidden_def (__nptl_setxid_sighandler)

static void
setxid_mark_thread (struct xid_command *cmdp, struct pthread *t)
{
  int ch;

  /* Wait until this thread is cloned.  */
  if (t->setxid_futex == -1
      && ! atomic_compare_and_exchange_bool_acq (&t->setxid_futex, -2, -1))
    do
      futex_wait_simple (&t->setxid_futex, -2, FUTEX_PRIVATE);
    while (t->setxid_futex == -2);

  /* Don't let the thread exit before the setxid handler runs.  */
  t->setxid_futex = 0;

  do
    {
      ch = t->cancelhandling;

      /* If the thread is exiting right now, ignore it.  */
      if ((ch & EXITING_BITMASK) != 0)
        {
          /* Release the futex if there is no other setxid in
             progress.  */
          if ((ch & SETXID_BITMASK) == 0)
            {
              t->setxid_futex = 1;
              futex_wake (&t->setxid_futex, 1, FUTEX_PRIVATE);
            }
          return;
        }
    }
  while (atomic_compare_and_exchange_bool_acq (&t->cancelhandling,
                                               ch | SETXID_BITMASK, ch));
}


static void
setxid_unmark_thread (struct xid_command *cmdp, struct pthread *t)
{
  int ch;

  do
    {
      ch = t->cancelhandling;
      if ((ch & SETXID_BITMASK) == 0)
        return;
    }
  while (atomic_compare_and_exchange_bool_acq (&t->cancelhandling,
                                               ch & ~SETXID_BITMASK, ch));

  /* Release the futex just in case.  */
  t->setxid_futex = 1;
  futex_wake (&t->setxid_futex, 1, FUTEX_PRIVATE);
}


static int
setxid_signal_thread (struct xid_command *cmdp, struct pthread *t)
{
  if ((t->cancelhandling & SETXID_BITMASK) == 0)
    return 0;

  int val;
  pid_t pid = __getpid ();
  val = INTERNAL_SYSCALL_CALL (tgkill, pid, t->tid, SIGSETXID);

  /* If this failed, it must have had not started yet or else exited.  */
  if (!INTERNAL_SYSCALL_ERROR_P (val))
    {
      atomic_increment (&cmdp->cntr);
      return 1;
    }
  else
    return 0;
}

int
attribute_hidden
__nptl_setxid (struct xid_command *cmdp)
{
  int signalled;
  int result;
  lll_lock (GL (dl_stack_cache_lock), LLL_PRIVATE);

  xidcmd = cmdp;
  cmdp->cntr = 0;
  cmdp->error = -1;

  struct pthread *self = THREAD_SELF;

  /* Iterate over the list with system-allocated threads first.  */
  list_t *runp;
  list_for_each (runp, &GL (dl_stack_used))
    {
      struct pthread *t = list_entry (runp, struct pthread, list);
      if (t == self)
        continue;

      setxid_mark_thread (cmdp, t);
    }

  /* Now the list with threads using user-allocated stacks.  */
  list_for_each (runp, &GL (dl_stack_user))
    {
      struct pthread *t = list_entry (runp, struct pthread, list);
      if (t == self)
        continue;

      setxid_mark_thread (cmdp, t);
    }

  /* Iterate until we don't succeed in signalling anyone.  That means
     we have gotten all running threads, and their children will be
     automatically correct once started.  */
  do
    {
      signalled = 0;

      list_for_each (runp, &GL (dl_stack_used))
        {
          struct pthread *t = list_entry (runp, struct pthread, list);
          if (t == self)
            continue;

          signalled += setxid_signal_thread (cmdp, t);
        }

      list_for_each (runp, &GL (dl_stack_user))
        {
          struct pthread *t = list_entry (runp, struct pthread, list);
          if (t == self)
            continue;

          signalled += setxid_signal_thread (cmdp, t);
        }

      int cur = cmdp->cntr;
      while (cur != 0)
        {
          futex_wait_simple ((unsigned int *) &cmdp->cntr, cur,
                             FUTEX_PRIVATE);
          cur = cmdp->cntr;
        }
    }
  while (signalled != 0);

  /* Clean up flags, so that no thread blocks during exit waiting
     for a signal which will never come.  */
  list_for_each (runp, &GL (dl_stack_used))
    {
      struct pthread *t = list_entry (runp, struct pthread, list);
      if (t == self)
        continue;

      setxid_unmark_thread (cmdp, t);
    }

  list_for_each (runp, &GL (dl_stack_user))
    {
      struct pthread *t = list_entry (runp, struct pthread, list);
      if (t == self)
        continue;

      setxid_unmark_thread (cmdp, t);
    }

  /* This must be last, otherwise the current thread might not have
     permissions to send SIGSETXID syscall to the other threads.  */
  result = INTERNAL_SYSCALL_NCS (cmdp->syscall_no, 3,
                                 cmdp->id[0], cmdp->id[1], cmdp->id[2]);
  int error = 0;
  if (__glibc_unlikely (INTERNAL_SYSCALL_ERROR_P (result)))
    {
      error = INTERNAL_SYSCALL_ERRNO (result);
      __set_errno (error);
      result = -1;
    }
  setxid_error (cmdp, error);

  lll_unlock (GL (dl_stack_cache_lock), LLL_PRIVATE);
  return result;
}
