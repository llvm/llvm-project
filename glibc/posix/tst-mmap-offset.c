/* BZ #18877, BZ #21270, and BZ #24699 mmap offset test.

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

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <mmap_info.h>

#include <support/check.h>

static int fd;
static long int page_shift;
static char fname[] = "/tmp/tst-mmap-offset-XXXXXX";

static void
do_prepare (int argc, char **argv)
{
  fd = mkstemp64 (fname);
  if (fd < 0)
    FAIL_EXIT1 ("mkstemp failed");

  if (unlink (fname))
    FAIL_EXIT1 ("unlink failed");

  long sz = sysconf(_SC_PAGESIZE);
  if (sz == -1)
    sz = 4096L;
  page_shift = ffs (sz) - 1;
}

#define PREPARE do_prepare


/* Check if negative offsets are handled correctly by mmap.  */
static int
do_test_bz18877 (void)
{
  const int prot = PROT_READ | PROT_WRITE;
  const int flags = MAP_SHARED;
  const unsigned long length = 0x10000;
  const unsigned long offset = 0xace00000;
  const unsigned long size = offset + length;
  void *addr;

  if (ftruncate64 (fd, size))
    FAIL_RET ("ftruncate64 failed");

  addr = mmap (NULL, length, prot, flags, fd, offset);
  if (MAP_FAILED == addr)
    FAIL_RET ("mmap failed");

  /* This memcpy is likely to SIGBUS if mmap has messed up with offset.  */
  memcpy (addr, fname, sizeof (fname));

  return 0;
}

/* Check if invalid offset are handled correctly by mmap.  */
static int
do_test_large_offset (void)
{
  /* For architectures with sizeof (off_t) < sizeof (off64_t) mmap is
     implemented with __SYS_mmap2 syscall and the offset is represented in
     multiples of page size.  For offset larger than
     '1 << (page_shift + 8 * sizeof (off_t))' (that is, 1<<44 on system with
     page size of 4096 bytes) the system call silently truncates the offset.
     For this case glibc mmap implementation returns EINVAL.  */
  const int prot = PROT_READ | PROT_WRITE;
  const int flags = MAP_SHARED;
  const int64_t offset = 1ULL << (page_shift + 8 * sizeof (uint32_t));
  const size_t length = 4096;

  void *addr = mmap64 (NULL, length, prot, flags, fd, offset);
  if (mmap64_maximum_offset (page_shift) < UINT64_MAX)
    {
      if ((addr != MAP_FAILED) && (errno != EINVAL))
	FAIL_RET ("mmap succeed");
    }
  else
    {
      if (addr == MAP_FAILED)
	FAIL_RET ("mmap failed");
    }

  return 0;
}

int
do_test (void)
{
  int ret = 0;

  ret += do_test_bz18877 ();
  ret += do_test_large_offset ();

  return ret;
}

#include <support/test-driver.c>
