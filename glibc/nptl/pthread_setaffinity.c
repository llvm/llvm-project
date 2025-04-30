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
#include <pthreadP.h>
#include <sysdep.h>
#include <sys/types.h>
#include <shlib-compat.h>


int
__pthread_setaffinity_new (pthread_t th, size_t cpusetsize,
			   const cpu_set_t *cpuset)
{
  const struct pthread *pd = (const struct pthread *) th;
  int res;

  res = INTERNAL_SYSCALL_CALL (sched_setaffinity, pd->tid, cpusetsize,
			       cpuset);

  return (INTERNAL_SYSCALL_ERROR_P (res)
	  ? INTERNAL_SYSCALL_ERRNO (res)
	  : 0);
}
versioned_symbol (libc, __pthread_setaffinity_new,
		  pthread_setaffinity_np, GLIBC_2_34);

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_3_4, GLIBC_2_34)
compat_symbol (libpthread, __pthread_setaffinity_new,
	       pthread_setaffinity_np, GLIBC_2_3_4);
#endif

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_3_3, GLIBC_2_3_4)
int
__pthread_setaffinity_old (pthread_t th, cpu_set_t *cpuset)
{
  /* The old interface by default assumed a 1024 processor bitmap.  */
  return __pthread_setaffinity_new (th, 128, cpuset);
}
compat_symbol (libpthread, __pthread_setaffinity_old, pthread_setaffinity_np,
	       GLIBC_2_3_3);
#endif
