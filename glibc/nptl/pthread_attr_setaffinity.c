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
#include <stdlib.h>
#include <string.h>
#include <pthreadP.h>
#include <shlib-compat.h>


int
__pthread_attr_setaffinity_np (pthread_attr_t *attr, size_t cpusetsize,
			       const cpu_set_t *cpuset)
{
  struct pthread_attr *iattr;

  iattr = (struct pthread_attr *) attr;

  if (cpuset == NULL || cpusetsize == 0)
    {
      if (iattr->extension != NULL)
	{
	  free (iattr->extension->cpuset);
	  iattr->extension->cpuset = NULL;
	  iattr->extension->cpusetsize = 0;
	}
    }
  else
    {
      int ret = __pthread_attr_extension (iattr);
      if (ret != 0)
	return ret;

      if (iattr->extension->cpusetsize != cpusetsize)
	{
	  void *newp = realloc (iattr->extension->cpuset, cpusetsize);
	  if (newp == NULL)
	    return ENOMEM;

	  iattr->extension->cpuset = newp;
	  iattr->extension->cpusetsize = cpusetsize;
	}

      memcpy (iattr->extension->cpuset, cpuset, cpusetsize);
    }

  return 0;
}
libc_hidden_def (__pthread_attr_setaffinity_np)
versioned_symbol (libc, __pthread_attr_setaffinity_np,
		  pthread_attr_setaffinity_np, GLIBC_2_32);


#if SHLIB_COMPAT (libc, GLIBC_2_3_4, GLIBC_2_32)
/* Compat symbol with the old libc version.  */
strong_alias (__pthread_attr_setaffinity_np, __pthread_attr_setaffinity_alias)
compat_symbol (libc, __pthread_attr_setaffinity_alias,
	       pthread_attr_setaffinity_np, GLIBC_2_3_4);
#endif

#if SHLIB_COMPAT (libc, GLIBC_2_3_3, GLIBC_2_3_4)
int
__pthread_attr_setaffinity_old (pthread_attr_t *attr, cpu_set_t *cpuset)
{
  /* The old interface by default assumed a 1024 processor bitmap.  */
  return __pthread_attr_setaffinity_np (attr, 128, cpuset);
}
compat_symbol (libc, __pthread_attr_setaffinity_old,
	       pthread_attr_setaffinity_np, GLIBC_2_3_3);
#endif
