/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2003.

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
#include <limits.h>
#include <pthreadP.h>
#include <string.h>
#include <sysdep.h>
#include <sys/param.h>
#include <sys/types.h>
#include <shlib-compat.h>


int
__pthread_getaffinity_np (pthread_t th, size_t cpusetsize, cpu_set_t *cpuset)
{
  const struct pthread *pd = (const struct pthread *) th;

  int res = INTERNAL_SYSCALL_CALL (sched_getaffinity, pd->tid,
				   MIN (INT_MAX, cpusetsize), cpuset);
  if (INTERNAL_SYSCALL_ERROR_P (res))
    return INTERNAL_SYSCALL_ERRNO (res);

  /* Clean the rest of the memory the kernel didn't do.  */
  memset ((char *) cpuset + res, '\0', cpusetsize - res);

  return 0;
}
libc_hidden_def (__pthread_getaffinity_np)
versioned_symbol (libc, __pthread_getaffinity_np, pthread_getaffinity_np,
		  GLIBC_2_32);

#if SHLIB_COMPAT (libc, GLIBC_2_3_4, GLIBC_2_32)
strong_alias (__pthread_getaffinity_np, __pthread_getaffinity_alias)
compat_symbol (libc, __pthread_getaffinity_alias, pthread_getaffinity_np,
	       GLIBC_2_3_4);
#endif

#if SHLIB_COMPAT (libc, GLIBC_2_3_3, GLIBC_2_3_4)
int
__pthread_getaffinity_old (pthread_t th, cpu_set_t *cpuset)
{
  /* The old interface by default assumed a 1024 processor bitmap.  */
  return __pthread_getaffinity_np (th, 128, cpuset);
}
compat_symbol (libc, __pthread_getaffinity_old, pthread_getaffinity_np,
	       GLIBC_2_3_3);
#endif
