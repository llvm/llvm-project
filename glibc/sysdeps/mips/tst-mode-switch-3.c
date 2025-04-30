/* Copyright (C) 2014-2021 Free Software Foundation, Inc.
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
#include <unistd.h>
#include <setjmp.h>
#include <sys/prctl.h>

#if __mips_fpr != 0 || _MIPS_SPFPSET != 16
# error This test requires -mfpxx -mno-odd-spreg
#endif

/* This test verifies that mode changes between a setjmp and longjmp do
   not corrupt the state of callee-saved registers.  */

static int mode[6] =
  {
    0,
    PR_FP_MODE_FR,
    PR_FP_MODE_FR | PR_FP_MODE_FRE,
    PR_FP_MODE_FR,
    0,
    PR_FP_MODE_FR | PR_FP_MODE_FRE
  };
static jmp_buf env;
float check1 = 2.0;
double check2 = 3.0;

static int
do_test (void)
{
  int i;
  int result = 0;

  for (i = 0 ; i < 7 ; i++)
    {
      int retval;
      register float test1 __asm ("$f20");
      register double test2 __asm ("$f22");

      /* Hide what we are doing to $f20 and $f22 from the compiler.  */
      __asm __volatile ("l.s %0,%2\n"
			"l.d %1,%3\n"
			: "=f" (test1), "=f" (test2)
			: "m" (check1), "m" (check2));

      retval = setjmp (env);

      /* Make sure the compiler knows we want to access the variables
         via the named registers again.  */
      __asm __volatile ("" : : "f" (test1), "f" (test2));

      if (test1 != check1 || test2 != check2)
	{
	  printf ("Corrupt register detected: $20 %f = %f, $22 %f = %f\n",
		  test1, check1, test2, check2);
	  result = 1;
	}

      if (retval == 0)
	{
	  if (prctl (PR_SET_FP_MODE, mode[i % 6]) != 0
	      && errno != ENOTSUP)
	    {
	      printf ("prctl PR_SET_FP_MODE failed: %m");
	      exit (1);
	    }
	  longjmp (env, 0);
	}
    }

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../../test-skeleton.c"
