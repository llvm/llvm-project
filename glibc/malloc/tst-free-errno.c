/* Test that free preserves errno.
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

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <support/check.h>
#include <support/support.h>
#include <support/temp_file.h>
#include <support/xunistd.h>

/* The __attribute__ ((weak)) prevents a GCC optimization.  Without
   it, GCC would "know" that errno is unchanged by calling free (ptr),
   when ptr was the result of a malloc call in the same function.  */
int __attribute__ ((weak))
get_errno (void)
{
  return errno;
}

static int
do_test (void)
{
  /* Check that free() preserves errno.  */
  {
    errno = 1789; /* Liberté, égalité, fraternité.  */
    free (NULL);
    TEST_VERIFY (get_errno () == 1789);
  }
  { /* Large memory allocations, to force mmap.  */
    enum { N = 2 };
    void * volatile ptrs[N];
    size_t i;
    for (i = 0; i < N; i++)
      ptrs[i] = xmalloc (5318153);
    for (i = 0; i < N; i++)
      {
        errno = 1789;
        free (ptrs[i]);
        TEST_VERIFY (get_errno () == 1789);
      }
  }

  /* Test a less common code path.
     When malloc() is based on mmap(), free() can sometimes call munmap().
     munmap() usually succeeds, but fails in a particular situation: when
       - it has to unmap the middle part of a VMA, and
       - the number of VMAs of a process is limited and the limit is
         already reached.
     The latter condition is fulfilled on Linux, when the file
     /proc/sys/vm/max_map_count exists.  For all known Linux versions
     the default limit is at most 65536.
   */
  #if defined __linux__
  if (xopen ("/proc/sys/vm/max_map_count", O_RDONLY, 0) >= 0)
    {
      /* Preparations.  */
      size_t pagesize = getpagesize ();
      void *firstpage_backup = xmalloc (pagesize);
      void *lastpage_backup = xmalloc (pagesize);
      /* Allocate a large memory area, as a bumper, so that the MAP_FIXED
         allocation later will not overwrite parts of the memory areas
         allocated to ld.so or libc.so.  */
      xmmap (NULL, 0x1000000, PROT_READ, MAP_ANONYMOUS | MAP_PRIVATE, -1);
      /* A file descriptor pointing to a regular file.  */
      int fd = create_temp_file ("tst-free-errno", NULL);
      if (fd < 0)
	FAIL_EXIT1 ("cannot create temporary file");

      /* Do a large memory allocation.  */
      size_t big_size = 0x1000000;
      void * volatile ptr = xmalloc (big_size - 0x100);
      char *ptr_aligned = (char *) ((uintptr_t) ptr & ~(pagesize - 1));
      /* This large memory allocation allocated a memory area
	 from ptr_aligned to ptr_aligned + big_size.
	 Enlarge this memory area by adding a page before and a page
	 after it.  */
      memcpy (firstpage_backup, ptr_aligned, pagesize);
      memcpy (lastpage_backup, ptr_aligned + big_size - pagesize,
	      pagesize);
      xmmap (ptr_aligned - pagesize, pagesize + big_size + pagesize,
	     PROT_READ | PROT_WRITE,
	     MAP_ANONYMOUS | MAP_PRIVATE | MAP_FIXED, -1);
      memcpy (ptr_aligned, firstpage_backup, pagesize);
      memcpy (ptr_aligned + big_size - pagesize, lastpage_backup,
	      pagesize);

      /* Now add as many mappings as we can.
	 Stop at 65536, in order not to crash the machine (in case the
	 limit has been increased by the system administrator).  */
      for (int i = 0; i < 65536; i++)
	if (mmap (NULL, pagesize, PROT_READ, MAP_FILE | MAP_PRIVATE, fd, 0)
	    == MAP_FAILED)
	  break;
      /* Now the number of VMAs of this process has hopefully attained
	 its limit.  */

      errno = 1789;
      /* This call to free() is supposed to call
	   munmap (ptr_aligned, big_size);
	 which increases the number of VMAs by 1, which is supposed
	 to fail.  */
      free (ptr);
      TEST_VERIFY (get_errno () == 1789);
    }
  #endif

  return 0;
}

#include <support/test-driver.c>
