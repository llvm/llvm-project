/* Tests register values retreived by getcontext() for mips o32.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ucontext.h>


#if !defined __mips__ || _MIPS_SIM != _ABIO32
# error "MIPS O32 specific test."
#endif

#define SP_REG 29

static int
do_test (void)
{
  ucontext_t ctx;
  memset (&ctx, 0, sizeof (ctx));
  int status = getcontext (&ctx);
  if (status)
    {
      printf ("\ngetcontext() failed, errno: %d.\n", errno);
      return 1;
    }

  if (ctx.uc_mcontext.gregs[SP_REG] == 0
      || ctx.uc_mcontext.gregs[SP_REG] > 0xffffffff)
    {
      printf ("\nError getcontext(): invalid $sp = 0x%llx.\n",
              ctx.uc_mcontext.gregs[SP_REG]);
      return 1;
    }

  if (ctx.uc_mcontext.pc == 0
      || ctx.uc_mcontext.pc > 0xffffffff)
    {
      printf ("\nError getcontext(): invalid ctx.uc_mcontext.pc = 0x%llx.\n",
              ctx.uc_mcontext.pc);
      return 1;
    }

  return 0;
}

#include <support/test-driver.c>
