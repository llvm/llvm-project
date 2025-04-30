/* Test CET property note parser.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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
#include <elf.h>
#include <tcb-offsets.h>

/* This test prints out "IBT" if Intel indirect branch tracking (IBT)
   is enabled at run-time, which is checked by tst-cet-property-2 to
   verify that the IBT violation is caught on IBT machines.  */

static int
do_test (void)
{
  unsigned int feature_1;
#ifdef __x86_64__
# define SEG_REG "fs"
#else
# define SEG_REG "gs"
#endif
  asm ("movl %%" SEG_REG ":%P1, %0"
       : "=r" (feature_1) : "i" (FEATURE_1_OFFSET));
  if ((feature_1 & GNU_PROPERTY_X86_FEATURE_1_IBT) != 0)
    printf ("IBT\n");

  return 0;
}

#include <support/test-driver.c>
