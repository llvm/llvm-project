/* Make sure that pthread_attr_getaffinity_np does not crash when the input
   cpuset size is smaller than that in the attribute structure.

   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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

#include <pthread.h>
#include <stdio.h>
#include <sched.h>
#include <errno.h>
#include <sys/param.h>


#define RETURN_IF_FAIL(f, ...) \
  ({									      \
    int ret = f (__VA_ARGS__);						      \
    if (ret != 0)							      \
      {									      \
	printf ("%s:%d: %s returned %d (errno = %d)\n", __FILE__, __LINE__,   \
		#f, ret, errno);					      \
	return ret;							      \
      }									      \
  })

static int
do_test (void)
{
  for (int i = 0; i < 10; i++)
    {
      pthread_attr_t attr;
      cpu_set_t *cpuset = CPU_ALLOC (512);
      size_t cpusetsize = CPU_ALLOC_SIZE (512);
      CPU_ZERO_S (cpusetsize, cpuset);

      RETURN_IF_FAIL (pthread_attr_init, &attr);
      RETURN_IF_FAIL (pthread_attr_setaffinity_np, &attr, cpusetsize, cpuset);
      CPU_FREE (cpuset);

      cpuset = CPU_ALLOC (1);
      cpusetsize = CPU_ALLOC_SIZE (1);
      RETURN_IF_FAIL (pthread_attr_getaffinity_np, &attr, cpusetsize, cpuset);
      CPU_FREE (cpuset);
    }
  return 0;
}


#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
