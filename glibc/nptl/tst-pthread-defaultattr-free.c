/* Test for user-after-free bug in pthread_getattr_default_np (bug 25999).
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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
#include <sched.h>
#include <support/check.h>
#include <support/xthread.h>

static int
do_test (void)
{
  /* This is a typical affinity size.  */
  enum { cpu_count = 128 };
  cpu_set_t *set = CPU_ALLOC (cpu_count);
  size_t set_size = CPU_ALLOC_SIZE (cpu_count);
  CPU_ZERO_S (set_size, set);
  CPU_SET (1, set);
  CPU_SET (3, set);

  /* Apply the affinity mask to the default attribute.  */
  pthread_attr_t attr;
  xpthread_attr_init (&attr);
  TEST_COMPARE (pthread_attr_setaffinity_np (&attr, set_size, set), 0);
  TEST_COMPARE (pthread_setattr_default_np (&attr), 0);
  xpthread_attr_destroy (&attr);

  /* Read back the default attribute and check affinity mask.  */
  pthread_getattr_default_np (&attr);
  CPU_ZERO_S (set_size, set);
  TEST_COMPARE (pthread_attr_getaffinity_np (&attr, set_size, set), 0);
  for (int i = 0; i < cpu_count; ++i)
    TEST_COMPARE (!!CPU_ISSET (i, set), i == 1 || i == 3);


  /* Apply a larger CPU affinity mask to the default attribute, to
     trigger reallocation.  */
  {
    cpu_set_t *large_set = CPU_ALLOC (4 * cpu_count);
    size_t large_set_size = CPU_ALLOC_SIZE (4 * cpu_count);
    CPU_ZERO_S (large_set_size, large_set);
    pthread_attr_t large_attr;
    xpthread_attr_init (&large_attr);
    TEST_COMPARE (pthread_attr_setaffinity_np (&large_attr,
                                               large_set_size, large_set), 0);
    TEST_COMPARE (pthread_setattr_default_np (&large_attr), 0);
    xpthread_attr_destroy (&large_attr);
    CPU_FREE (large_set);
  }

  /* Read back the default attribute and check affinity mask.  */
  CPU_ZERO_S (set_size, set);
  TEST_COMPARE (pthread_attr_getaffinity_np (&attr, set_size, set), 0);
  for (int i = 0; i < cpu_count; ++i)
    TEST_COMPARE (!!CPU_ISSET (i, set), i == 1 || i == 3);
  /* Due to bug 25999, the following caused a double-free abort.  */
  xpthread_attr_destroy (&attr);

  CPU_FREE (set);

  return 0;
}

#include <support/test-driver.c>
