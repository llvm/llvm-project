/* Check CET compatibility with legacy JIT engine via GLIBC_TUNABLES.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <support/test-driver.h>
#include <support/xunistd.h>

/* Check that mmapped legacy code won't trigger segfault with
   -fcf-protection and GLIBC_TUNABLES=glibc.cpu.hwcaps=-IBT,-SHSTK.  */

static int
do_test (void)
{
  void (*funcp) (void);
  funcp = xmmap (NULL, 0x1000, PROT_EXEC | PROT_READ | PROT_WRITE,
		 MAP_ANONYMOUS | MAP_PRIVATE, -1);
  printf ("mmap = %p\n", funcp);
  /* Write RET instruction.  */
  *(char *) funcp = 0xc3;
  funcp ();
  return EXIT_SUCCESS;
}

#include <support/test-driver.c>
