/* Test for syscall interfaces.
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

/* This test verifies that the x32 system call handling zero-extends
   unsigned 32-bit arguments to the 64-bit argument registers for
   system calls (bug 25810).  The bug is specific to x32, but the test
   should pass on all architectures.  */

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <support/check.h>
#include <support/xunistd.h>

/* On x32, this can be passed in a single 64-bit integer register.  */
struct Array
{
  size_t length;
  void *ptr;
};

static int error_count;

__attribute__ ((noclone, noinline))
struct Array
allocate (size_t bytes)
{
  if (!bytes)
    return __extension__ (struct Array) {0, 0};

  void *p = mmap (0x0, bytes, PROT_READ | PROT_WRITE,
		  MAP_PRIVATE | MAP_ANON, -1, 0);
  if (p == MAP_FAILED)
    return __extension__ (struct Array) {0, 0};

  return __extension__ (struct Array) {bytes, p};
}

__attribute__ ((noclone, noinline))
void
deallocate (struct Array b)
{
  /* On x32, the 64-bit integer register containing `b' may be copied
     to another 64-bit integer register to pass the second argument to
     munmap.  */
  if (b.length && munmap (b.ptr, b.length))
    {
      printf ("munmap error: %m\n");
      error_count++;
    }
}

__attribute__ ((noclone, noinline))
void *
do_mmap (void *addr, size_t length)
{
  return mmap (addr, length, PROT_READ | PROT_WRITE,
	       MAP_PRIVATE | MAP_ANON, -1, 0);
}

__attribute__ ((noclone, noinline))
void *
reallocate (struct Array b)
{
  /* On x32, the 64-bit integer register containing `b' may be copied
     to another 64-bit integer register to pass the second argument to
     do_mmap.  */
  if (b.length)
    return do_mmap (b.ptr, b.length);
  return NULL;
}

__attribute__ ((noclone, noinline))
void
protect (struct Array b)
{
  if (b.length)
    {
      /* On x32, the 64-bit integer register containing `b' may be copied
	 to another 64-bit integer register to pass the second argument
	 to mprotect.  */
      if (mprotect (b.ptr, b.length,
		    PROT_READ | PROT_WRITE | PROT_EXEC))
	{
	  printf ("mprotect error: %m\n");
	  error_count++;
	}
    }
}

__attribute__ ((noclone, noinline))
ssize_t
do_read (int fd, void *ptr, struct Array b)
{
  /* On x32, the 64-bit integer register containing `b' may be copied
     to another 64-bit integer register to pass the second argument to
     read.  */
  if (b.length)
    return read (fd, ptr, b.length);
  return 0;
}

__attribute__ ((noclone, noinline))
ssize_t
do_write (int fd, void *ptr, struct Array b)
{
  /* On x32, the 64-bit integer register containing `b' may be copied
     to another 64-bit integer register to pass the second argument to
     write.  */
  if (b.length)
    return write (fd, ptr, b.length);
  return 0;
}

static int
do_test (void)
{
  struct Array array;

  array = allocate (1);
  protect (array);
  deallocate (array);
  void *p = reallocate (array);
  if (p == MAP_FAILED)
    {
      printf ("mmap error: %m\n");
      error_count++;
    }
  array.ptr = p;
  protect (array);
  deallocate (array);

  int fd = xopen ("/dev/null", O_RDWR, 0);
  char buf[2];
  array.ptr = buf;
  if (do_read (fd, array.ptr, array) == -1)
    {
      printf ("read error: %m\n");
      error_count++;
    }
  if (do_write (fd, array.ptr, array) == -1)
    {
      printf ("write error: %m\n");
      error_count++;
    }
  xclose (fd);

  return error_count ? EXIT_FAILURE : EXIT_SUCCESS;
}

#include <support/test-driver.c>
