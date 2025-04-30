/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Andreas Jaeger <aj@suse.de>, 2000.

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

#include <fpu_control.h>
#include <stdio.h>

#ifndef FPU_CONTROL
# define FPU_CONTROL _FPU_DEFAULT
#endif

static int
do_test (void)
{
#ifdef _FPU_GETCW
/* Some architectures don't have _FPU_GETCW (e.g. Linux/Alpha).  */
  fpu_control_t cw;

  _FPU_GETCW (cw);

  cw &= ~_FPU_RESERVED;

  if (cw != (FPU_CONTROL & ~_FPU_RESERVED))
    printf ("control word is 0x%lx but should be 0x%lx.\n",
	    (long int) cw, (long int) (FPU_CONTROL & ~_FPU_RESERVED));

  return cw != (FPU_CONTROL & ~_FPU_RESERVED);

#else
  return 0;
#endif
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
