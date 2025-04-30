/* pthread_setname_np -- Set  thread name.  Linux version
   Copyright (C) 2010-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

#include <errno.h>
#include <fcntl.h>
#include <pthreadP.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/prctl.h>

#include <not-cancel.h>


int
__pthread_setname_np (pthread_t th, const char *name)
{
  const struct pthread *pd = (const struct pthread *) th;

  /* Unfortunately the kernel headers do not export the TASK_COMM_LEN
     macro.  So we have to define it here.  */
#define TASK_COMM_LEN 16
  size_t name_len = strlen (name);
  if (name_len >= TASK_COMM_LEN)
    return ERANGE;

  if (pd == THREAD_SELF)
    return __prctl (PR_SET_NAME, name) ? errno : 0;

#define FMT "/proc/self/task/%u/comm"
  char fname[sizeof (FMT) + 8];
  sprintf (fname, FMT, (unsigned int) pd->tid);

  int fd = __open64_nocancel (fname, O_RDWR);
  if (fd == -1)
    return errno;

  int res = 0;
  ssize_t n = TEMP_FAILURE_RETRY (__write_nocancel (fd, name, name_len));
  if (n < 0)
    res = errno;
  else if (n != name_len)
    res = EIO;

  __close_nocancel_nostatus (fd);

  return res;
}
versioned_symbol (libc, __pthread_setname_np, pthread_setname_np,
                  GLIBC_2_34);

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_12, GLIBC_2_34)
compat_symbol (libpthread, __pthread_setname_np, pthread_setname_np,
               GLIBC_2_12);
#endif
