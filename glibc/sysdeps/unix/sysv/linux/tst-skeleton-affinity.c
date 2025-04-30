/* Generic test case for CPU affinity functions.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

/* This file is included by the tst-affinity*.c files to test the two
   variants of the functions, under different conditions.  The
   following functions have to be definied:

   static int getaffinity (size_t, cpu_set_t *);
   static int setaffinity (size_t, const cpu_set_t *);
   static bool early_test (struct conf *);

   The first two functions shall affect the affinity mask for the
   current thread and return 0 for success, -1 for error (with an
   error code in errno).

   early_test is invoked before the tests in this file affect the
   affinity masks.  If it returns true, testing continues, otherwise
   no more tests run and the overall test exits with status 1.
*/

#include <errno.h>
#include <limits.h>
#include <sched.h>
#include <stdbool.h>
#include <stdio.h>

/* CPU set configuration determined.  Can be used from early_test.  */
struct conf
{
  int set_size;			/* in bits */
  int last_cpu;
};

static int
find_set_size (void)
{
  /* There is considerable controversy about how to determine the size
     of the kernel CPU mask.  The probing loop below is only intended
     for testing purposes.  */
  for (int num_cpus = 64; num_cpus <= INT_MAX / 2; ++num_cpus)
    {
      cpu_set_t *set = CPU_ALLOC (num_cpus);
      size_t size = CPU_ALLOC_SIZE (num_cpus);

      if (set == NULL)
	{
	  printf ("error: CPU_ALLOC (%d) failed\n", num_cpus);
	  return -1;
	}
      if (getaffinity (size, set) == 0)
	{
	  CPU_FREE (set);
	  return num_cpus;
	}
      if (errno != EINVAL)
	{
	  printf ("error: getaffinity for %d CPUs: %m\n", num_cpus);
	  CPU_FREE (set);
	  return -1;
	}
      CPU_FREE (set);
    }
  puts ("error: Cannot find maximum CPU number");
  return -1;
}

static int
find_last_cpu (const cpu_set_t *set, size_t size)
{
  /* We need to determine the set size with CPU_COUNT_S and the
     cpus_found counter because there is no direct way to obtain the
     actual CPU set size, in bits, from the value of
     CPU_ALLOC_SIZE.  */
  size_t cpus_found = 0;
  size_t total_cpus = CPU_COUNT_S (size, set);
  int last_cpu = -1;

  for (int cpu = 0; cpus_found < total_cpus; ++cpu)
    {
      if (CPU_ISSET_S (cpu, size, set))
	{
	  last_cpu = cpu;
	  ++cpus_found;
	}
    }
  return last_cpu;
}

static void
setup_conf (struct conf *conf)
{
  *conf = (struct conf) {-1, -1};
  conf->set_size = find_set_size ();
  if (conf->set_size > 0)
    {
      cpu_set_t *set = CPU_ALLOC (conf->set_size);

      if (set == NULL)
	{
	  printf ("error: CPU_ALLOC (%d) failed\n", conf->set_size);
	  CPU_FREE (set);
	  return;
	}
      if (getaffinity (CPU_ALLOC_SIZE (conf->set_size), set) < 0)
	{
	  printf ("error: getaffinity failed: %m\n");
	  CPU_FREE (set);
	  return;
	}
      conf->last_cpu = find_last_cpu (set, CPU_ALLOC_SIZE (conf->set_size));
      if (conf->last_cpu < 0)
	puts ("info: No test CPU found");
      CPU_FREE (set);
    }
}

static bool
test_size (const struct conf *conf, size_t size)
{
  if (size < conf->set_size)
    {
      printf ("info: Test not run for CPU set size %zu\n", size);
      return true;
    }

  cpu_set_t *initial_set = CPU_ALLOC (size);
  cpu_set_t *set2 = CPU_ALLOC (size);
  cpu_set_t *active_cpu_set = CPU_ALLOC (size);

  if (initial_set == NULL || set2 == NULL || active_cpu_set == NULL)
    {
      printf ("error: size %zu: CPU_ALLOC failed\n", size);
      return false;
    }
  size_t kernel_size = CPU_ALLOC_SIZE (size);

  if (getaffinity (kernel_size, initial_set) < 0)
    {
      printf ("error: size %zu: getaffinity: %m\n", size);
      return false;
    }
  if (setaffinity (kernel_size, initial_set) < 0)
    {
      printf ("error: size %zu: setaffinity: %m\n", size);
      return true;
    }

  /* Use one-CPU set to test switching between CPUs.  */
  int last_active_cpu = -1;
  for (int cpu = 0; cpu <= conf->last_cpu; ++cpu)
    {
      int active_cpu = sched_getcpu ();
      if (last_active_cpu >= 0 && last_active_cpu != active_cpu)
	{
	  printf ("error: Unexpected CPU %d, expected %d\n",
		  active_cpu, last_active_cpu);
	  return false;
	}

      if (!CPU_ISSET_S (cpu, kernel_size, initial_set))
	continue;
      last_active_cpu = cpu;

      CPU_ZERO_S (kernel_size, active_cpu_set);
      CPU_SET_S (cpu, kernel_size, active_cpu_set);
      if (setaffinity (kernel_size, active_cpu_set) < 0)
	{
	  printf ("error: size %zu: setaffinity (%d): %m\n", size, cpu);
	  return false;
	}
      active_cpu = sched_getcpu ();
      if (active_cpu != cpu)
	{
	  printf ("error: Unexpected CPU %d, expected %d\n", active_cpu, cpu);
	  return false;
	}
      unsigned int numa_cpu, numa_node;
      if (getcpu (&numa_cpu, &numa_node) != 0)
	{
	  printf ("error: getcpu: %m\n");
	  return false;
	}
      if ((unsigned int) active_cpu != numa_cpu)
	{
	  printf ("error: Unexpected CPU %d, expected %d\n",
		  active_cpu, numa_cpu);
	  return false;
	}
      if (getaffinity (kernel_size, set2) < 0)
	{
	  printf ("error: size %zu: getaffinity (2): %m\n", size);
	  return false;
	}
      if (!CPU_EQUAL_S (kernel_size, active_cpu_set, set2))
	{
	  printf ("error: size %zu: CPU sets do not match\n", size);
	  return false;
	}
    }

  /* Test setting the all-ones set.  */
  for (int cpu = 0; cpu < size; ++cpu)
    CPU_SET_S (cpu, kernel_size, set2);
  if (setaffinity (kernel_size, set2) < 0)
    {
      printf ("error: size %zu: setaffinity (3): %m\n", size);
      return false;
    }

  if (setaffinity (kernel_size, initial_set) < 0)
    {
      printf ("error: size %zu: setaffinity (4): %m\n", size);
      return false;
    }
  if (getaffinity (kernel_size, set2) < 0)
    {
      printf ("error: size %zu: getaffinity (3): %m\n", size);
      return false;
    }
  if (!CPU_EQUAL_S (kernel_size, initial_set, set2))
    {
      printf ("error: size %zu: CPU sets do not match (2)\n", size);
      return false;
    }

  CPU_FREE (initial_set);
  CPU_FREE (set2);
  CPU_FREE (active_cpu_set);

  return true;
}

static int
do_test (void)
{
  {
    cpu_set_t set;
    if (getaffinity (sizeof (set), &set) < 0 && errno == ENOSYS)
      {
	puts ("warning: getaffinity not supported, test cannot run");
	return 0;
      }
    if (sched_getcpu () < 0 && errno == ENOSYS)
      {
	puts ("warning: sched_getcpu not supported, test cannot run");
	return 0;
      }
  }

  struct conf conf;
  setup_conf (&conf);
  /* Note: The CPU set size in bits can be less than the CPU count
     (and the maximum test CPU) because the userspace interface rounds
     up the bit count, and the rounded-up buffer size is passed into
     the kernel.  The kernel does not know that some of the buffer are
     actually padding, and writes data there.  */
  printf ("info: Detected CPU set size (in bits): %d\n", conf.set_size);
  printf ("info: Maximum test CPU: %d\n", conf.last_cpu);
  if (conf.set_size < 0 || conf.last_cpu < 0)
    return 1;

  if (!early_test (&conf))
    return 1;

  bool error = false;
  error |= !test_size (&conf, 1024);
  error |= !test_size (&conf, conf.set_size);
  error |= !test_size (&conf, 2);
  error |= !test_size (&conf, 32);
  error |= !test_size (&conf, 40);
  error |= !test_size (&conf, 64);
  error |= !test_size (&conf, 96);
  error |= !test_size (&conf, 128);
  error |= !test_size (&conf, 256);
  error |= !test_size (&conf, 8192);
  return error;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
