/* Determine various system internal values, Linux version.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1996.

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

#include <array_length.h>
#include <dirent.h>
#include <errno.h>
#include <ldsodefs.h>
#include <limits.h>
#include <not-cancel.h>
#include <stdio.h>
#include <stdio_ext.h>
#include <sys/mman.h>
#include <sys/sysinfo.h>
#include <sysdep.h>

/* Compute the population count of the entire array.  */
static int
__get_nprocs_count (const unsigned long int *array, size_t length)
{
  int count = 0;
  for (size_t i = 0; i < length; ++i)
    if (__builtin_add_overflow (count,  __builtin_popcountl (array[i]),
				&count))
      return INT_MAX;
  return count;
}

/* __get_nprocs with a large buffer.  */
static int
__get_nprocs_large (void)
{
  /* This code cannot use scratch_buffer because it is used during
     malloc initialization.  */
  size_t pagesize = GLRO (dl_pagesize);
  unsigned long int *page = __mmap (0, pagesize, PROT_READ | PROT_WRITE,
				    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  if (page == MAP_FAILED)
    return 2;
  int r = INTERNAL_SYSCALL_CALL (sched_getaffinity, 0, pagesize, page);
  int count;
  if (r > 0)
    count = __get_nprocs_count (page, pagesize / sizeof (unsigned long int));
  else if (r == -EINVAL)
    /* One page is still not enough to store the bits.  A more-or-less
       arbitrary value.  This assumes t hat such large systems never
       happen in practice.  */
    count = GLRO (dl_pagesize) * CHAR_BIT;
  else
    count = 2;
  __munmap (page, GLRO (dl_pagesize));
  return count;
}

int
__get_nprocs (void)
{
  /* Fast path for most systems.  The kernel expects a buffer size
     that is a multiple of 8.  */
  unsigned long int small_buffer[1024 / CHAR_BIT / sizeof (unsigned long int)];
  int r = INTERNAL_SYSCALL_CALL (sched_getaffinity, 0,
				 sizeof (small_buffer), small_buffer);
  if (r > 0)
    return __get_nprocs_count (small_buffer, r / sizeof (unsigned long int));
  else if (r == -EINVAL)
    /* The kernel requests a larger buffer to store the data.  */
    return __get_nprocs_large ();
  else
    /* Some other error.  2 is conservative (not a uniprocessor
       system, so atomics are needed). */
    return 2;
}
libc_hidden_def (__get_nprocs)
weak_alias (__get_nprocs, get_nprocs)


/* On some architectures it is possible to distinguish between configured
   and active cpus.  */
int
__get_nprocs_conf (void)
{
  /* Try to use the sysfs filesystem.  It has actual information about
     online processors.  */
  DIR *dir = __opendir ("/sys/devices/system/cpu");
  if (dir != NULL)
    {
      int count = 0;
      struct dirent64 *d;

      while ((d = __readdir64 (dir)) != NULL)
	/* NB: the sysfs has d_type support.  */
	if (d->d_type == DT_DIR && strncmp (d->d_name, "cpu", 3) == 0)
	  {
	    char *endp;
	    unsigned long int nr = strtoul (d->d_name + 3, &endp, 10);
	    if (nr != ULONG_MAX && endp != d->d_name + 3 && *endp == '\0')
	      ++count;
	  }

      __closedir (dir);

      return count;
    }

  return 1;
}
libc_hidden_def (__get_nprocs_conf)
weak_alias (__get_nprocs_conf, get_nprocs_conf)


/* Compute (num*mem_unit)/pagesize, but avoid overflowing long int.
   In practice, mem_unit is never bigger than the page size, so after
   the first loop it is 1.  [In the kernel, it is initialized to
   PAGE_SIZE in mm/page_alloc.c:si_meminfo(), and then in
   kernel.sys.c:do_sysinfo() it is set to 1 if unsigned long can
   represent all the sizes measured in bytes].  */
static long int
sysinfo_mempages (unsigned long int num, unsigned int mem_unit)
{
  unsigned long int ps = __getpagesize ();

  while (mem_unit > 1 && ps > 1)
    {
      mem_unit >>= 1;
      ps >>= 1;
    }
  num *= mem_unit;
  while (ps > 1)
    {
      ps >>= 1;
      num >>= 1;
    }
  return num;
}

/* Return the number of pages of total/available physical memory in
   the system.  This used to be done by parsing /proc/meminfo, but
   that's unnecessarily expensive (and /proc is not always available).
   The sysinfo syscall provides the same information, and has been
   available at least since kernel 2.3.48.  */
long int
__get_phys_pages (void)
{
  struct sysinfo info;

  __sysinfo (&info);
  return sysinfo_mempages (info.totalram, info.mem_unit);
}
libc_hidden_def (__get_phys_pages)
weak_alias (__get_phys_pages, get_phys_pages)

long int
__get_avphys_pages (void)
{
  struct sysinfo info;

  __sysinfo (&info);
  return sysinfo_mempages (info.freeram, info.mem_unit);
}
libc_hidden_def (__get_avphys_pages)
weak_alias (__get_avphys_pages, get_avphys_pages)
