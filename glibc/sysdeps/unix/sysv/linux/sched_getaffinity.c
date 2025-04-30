/* Copyright (C) 2002-2021 Free Software Foundation, Inc.
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
#include <sched.h>
#include <string.h>
#include <sysdep.h>
#include <sys/param.h>
#include <sys/types.h>
#include <shlib-compat.h>


#if SHLIB_COMPAT (libc, GLIBC_2_3_3, GLIBC_2_3_4)
extern int __sched_getaffinity_new (pid_t, size_t, cpu_set_t *);
libc_hidden_proto (__sched_getaffinity_new)
#endif

int
__sched_getaffinity_new (pid_t pid, size_t cpusetsize, cpu_set_t *cpuset)
{
  int res = INLINE_SYSCALL (sched_getaffinity, 3, pid,
			    MIN (INT_MAX, cpusetsize), cpuset);
  if (res != -1)
    {
      /* Clean the rest of the memory the kernel didn't do.  */
      memset ((char *) cpuset + res, '\0', cpusetsize - res);

      res = 0;
    }
  return res;
}
versioned_symbol (libc, __sched_getaffinity_new, sched_getaffinity,
		  GLIBC_2_3_4);


#if SHLIB_COMPAT (libc, GLIBC_2_3_3, GLIBC_2_3_4)
libc_hidden_def (__sched_getaffinity_new)

int
attribute_compat_text_section
__sched_getaffinity_old (pid_t pid, cpu_set_t *cpuset)
{
  /* The old interface by default assumed a 1024 processor bitmap.  */
  return __sched_getaffinity_new (pid, 128, cpuset);
}
compat_symbol (libc, __sched_getaffinity_old, sched_getaffinity, GLIBC_2_3_3);
#endif
