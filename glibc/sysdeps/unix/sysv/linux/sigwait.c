/* Copyright (C) 1997-2021 Free Software Foundation, Inc.
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

#include <signal.h>
#include <sysdep-cancel.h>
#include <errno.h>

int
__sigwait (const sigset_t *set, int *sig)
{
  siginfo_t si;
  int ret;
  do
    ret = __sigtimedwait (set, &si, 0);
  /* Applications do not expect sigwait to return with EINTR, and the
     error code is not specified by POSIX.  */
  while (ret < 0 && errno == EINTR);
  if (ret < 0)
    return errno;
  *sig = si.si_signo;
  return 0;
}
libc_hidden_def (__sigwait)
weak_alias (__sigwait, sigwait)
strong_alias (__sigwait, __libc_sigwait)
