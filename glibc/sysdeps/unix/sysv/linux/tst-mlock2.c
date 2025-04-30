/* Test the mlock2 function.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.

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
#include <stdio.h>
#include <support/check.h>
#include <support/xunistd.h>
#include <sys/mman.h>

/* Allocate a page using mmap.  */
static void *
get_page (void)
{
  return xmmap (NULL, 1, PROT_READ | PROT_WRITE,
                MAP_ANONYMOUS | MAP_PRIVATE, -1);
}

static int
do_test (void)
{
  /* Current kernels have a small reserve of locked memory, so this
     test does not need any privileges to run.  */

  void *page = get_page ();
  if (mlock (page, 1) != 0)
    FAIL_EXIT1 ("mlock: %m\n");
  xmunmap (page, 1);

  page = get_page ();
  if (mlock2 (page, 1, 0) != 0)
    /* Should be implemented using mlock if necessary.  */
    FAIL_EXIT1 ("mlock2 (0): %m\n");
  xmunmap (page, 1);

  page = get_page ();
  int ret = mlock2 (page, 1, MLOCK_ONFAULT);
  if (ret != 0)
    {
      TEST_VERIFY (ret == -1);
      if (errno != EINVAL)
        /* EINVAL means the system does not support the mlock2 system
           call.  */
        FAIL_EXIT1 ("mlock2 (0): %m\n");
      else
        puts ("warning: mlock2 system call not supported");
    }
  xmunmap (page, 1);

  return 0;
}

#include <support/test-driver.c>
