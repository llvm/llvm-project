/* Special use of signals internally.  Stub version.
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
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef __INTERNAL_SIGNALS_H
# define __INTERNAL_SIGNALS_H

#include <signal.h>
#include <sigsetops.h>
#include <stdbool.h>
#include <stddef.h>

/* How many signal numbers need to be reserved for libpthread's private uses
   (SIGCANCEL and SIGSETXID).  */
#define RESERVED_SIGRT  0

static inline bool
__is_internal_signal (int sig)
{
  return false;
}

static inline void
__clear_internal_signals (sigset_t *set)
{
}

static inline void
__libc_signal_block_all (sigset_t *set)
{
  sigset_t allset;
  __sigfillset (&allset);
  __sigprocmask (SIG_BLOCK, &allset, set);
}

static inline void
__libc_signal_block_app (sigset_t *set)
{
  sigset_t allset;
  __sigfillset (&allset);
  __clear_internal_signals (&allset);
  __sigprocmask (SIG_BLOCK, &allset, set);
}

/* Restore current process signal mask.  */
static inline void
__libc_signal_restore_set (const sigset_t *set)
{
  __sigprocmask (SIG_SETMASK, set, NULL);
}


#endif /* __INTERNAL_SIGNALS_H  */
