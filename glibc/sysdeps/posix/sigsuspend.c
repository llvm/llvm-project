/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <signal.h>
#include <stddef.h>
#include <unistd.h>


/* Change the set of blocked signals to SET,
   wait until a signal arrives, and restore the set of blocked signals.  */
int
__sigsuspend (const sigset_t *set)
{
  sigset_t oset;
  int save;

  if (set == NULL)
    {
      __set_errno (EINVAL);
      return -1;
    }

  if (sigprocmask (SIG_SETMASK, set, &oset) < 0)
    return -1;

  (void) pause();
  save = errno;

  if (sigprocmask (SIG_SETMASK, &oset, (sigset_t *) NULL) < 0)
    return -1;

  __set_errno (save);
  return -1;
}
libc_hidden_def (__sigsuspend)
weak_alias (__sigsuspend, sigsuspend)
