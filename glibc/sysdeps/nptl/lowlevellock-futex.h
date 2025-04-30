/* Low-level locking access to futex facilities.  Stub version.
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef _LOWLEVELLOCK_FUTEX_H
#define _LOWLEVELLOCK_FUTEX_H   1

#include <_ns_unmigratable.h>

#ifndef __ASSEMBLER__
# include <sysdep.h>
# include <sysdep-cancel.h>
# include <kernel-features.h>
#endif

#define FUTEX_WAIT		0
#define FUTEX_WAKE		1
#define FUTEX_REQUEUE		3
#define FUTEX_CMP_REQUEUE	4
#define FUTEX_WAKE_OP		5
#define FUTEX_OP_CLEAR_WAKE_IF_GT_ONE	((4 << 24) | 1)
#define FUTEX_LOCK_PI		6
#define FUTEX_UNLOCK_PI		7
#define FUTEX_TRYLOCK_PI	8
#define FUTEX_WAIT_BITSET	9
#define FUTEX_WAKE_BITSET	10
#define FUTEX_WAIT_REQUEUE_PI   11
#define FUTEX_CMP_REQUEUE_PI    12
#define FUTEX_PRIVATE_FLAG	128
#define FUTEX_CLOCK_REALTIME	256

#define FUTEX_BITSET_MATCH_ANY	0xffffffff

/* Values for 'private' parameter of locking macros.  Yes, the
   definition seems to be backwards.  But it is not.  The bit will be
   reversed before passing to the system call.  */
#define LLL_PRIVATE	0
#define LLL_SHARED	FUTEX_PRIVATE_FLAG

#ifndef __ASSEMBLER__
# define __lll_private_flag(fl, private) \
  (((fl) | FUTEX_PRIVATE_FLAG) ^ (private))

# define lll_futex_syscall(nargs, futexp, op, ...)                      \
  ({                                                                    \
    __try_to_mark_as_unmigratable(futexp);                              \
    long int __ret = INTERNAL_SYSCALL (futex, nargs, futexp, op, 	\
				       __VA_ARGS__);                    \
    (__glibc_unlikely (INTERNAL_SYSCALL_ERROR_P (__ret))         	\
     ? -INTERNAL_SYSCALL_ERRNO (__ret) : 0);                     	\
  })

/* For most of these macros, the return value is never really used.
   Nevertheless, the protocol is that each one returns a negated errno
   code for failure or zero for success.  (Note that the corresponding
   Linux system calls can sometimes return positive values for success
   cases too.  We never use those values.)  */


/* Wait while *FUTEXP == VAL for an lll_futex_wake call on FUTEXP.  */
# define lll_futex_wait(futexp, val, private) \
  lll_futex_timed_wait (futexp, val, NULL, private)

# define lll_futex_timed_wait(futexp, val, timeout, private)     \
  lll_futex_syscall (4, futexp,                                 \
		     __lll_private_flag (FUTEX_WAIT, private),  \
		     val, timeout)

/* Verify whether the supplied clockid is supported by
   lll_futex_clock_wait_bitset.  */
# define lll_futex_supported_clockid(clockid)			\
  ((clockid) == CLOCK_REALTIME || (clockid) == CLOCK_MONOTONIC)

/* Wake up up to NR waiters on FUTEXP.  */
# define lll_futex_wake(futexp, nr, private)                             \
  lll_futex_syscall (4, futexp,                                         \
		     __lll_private_flag (FUTEX_WAKE, private), nr, 0)

/* Wake up up to NR_WAKE waiters on FUTEXP.  Move up to NR_MOVE of the
   rest from waiting on FUTEXP to waiting on MUTEX (a different futex).
   Returns non-zero if error happened, zero if success.  */
# define lll_futex_requeue(futexp, nr_wake, nr_move, mutex, val, private) \
  lll_futex_syscall (6, futexp,                                         \
		     __lll_private_flag (FUTEX_CMP_REQUEUE, private),   \
		     nr_wake, nr_move, mutex, val)

/* Wake up up to NR_WAKE waiters on FUTEXP and NR_WAKE2 on FUTEXP2.
   Returns non-zero if error happened, zero if success.  */
# define lll_futex_wake_unlock(futexp, nr_wake, nr_wake2, futexp2, private) \
  lll_futex_syscall (6, futexp,                                         \
		     __lll_private_flag (FUTEX_WAKE_OP, private),       \
		     nr_wake, nr_wake2, futexp2,                        \
		     FUTEX_OP_CLEAR_WAKE_IF_GT_ONE)


#define lll_futex_timed_unlock_pi(futexp, private) 			\
  lll_futex_syscall (4, futexp,						\
		     __lll_private_flag (FUTEX_UNLOCK_PI, private),	\
		     0, 0)

/* Like lll_futex_requeue, but pairs with lll_futex_wait_requeue_pi
   and inherits priority from the waiter.  */
# define lll_futex_cmp_requeue_pi(futexp, nr_wake, nr_move, mutex,       \
                                 val, private)                          \
  lll_futex_syscall (6, futexp,                                         \
		     __lll_private_flag (FUTEX_CMP_REQUEUE_PI,          \
					 private),                      \
		     nr_wake, nr_move, mutex, val)

/* Like lll_futex_wait, but acting as a cancellable entrypoint.  */
# define lll_futex_wait_cancel(futexp, val, private) \
  ({                                                                   \
    int __oldtype = LIBC_CANCEL_ASYNC ();			       \
    long int __err = lll_futex_wait (futexp, val, LLL_SHARED);	       \
    LIBC_CANCEL_RESET (__oldtype);				       \
    __err;							       \
  })

/* Like lll_futex_timed_wait, but acting as a cancellable entrypoint.  */
# define lll_futex_timed_wait_cancel(futexp, val, timeout, private) \
  ({									   \
    int __oldtype = LIBC_CANCEL_ASYNC ();			       	   \
    long int __err = lll_futex_timed_wait (futexp, val, timeout, private); \
    LIBC_CANCEL_RESET (__oldtype);					   \
    __err;								   \
  })

#endif  /* !__ASSEMBLER__  */

#endif  /* lowlevellock-futex.h */
