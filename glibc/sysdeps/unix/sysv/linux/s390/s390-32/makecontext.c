/* Copyright (C) 2001-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Martin Schwidefsky (schwidefsky@de.ibm.com).

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

#include <libintl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <ucontext.h>

/* This implementation can handle any ARGC value but only
   normal integer type parameters. Parameters of type float,
   double, complex and structure with sizes 0, 2, 4 or 8
   won't work.
   makecontext sets up a stack and the registers for the
   user context. The stack looks like this:
	   size                         offset
    %r15 ->    +-----------------------+
	     4 | back chain (zero)     |  0
	     4 | reserved              |  4
	    88 | save area for (*func) |  8
	       +-----------------------+
	     n | overflow parameters   | 96
	       +-----------------------+
   The registers are set up like this:
     %r2-%r6: parameters 1 to 5
     %r7    : (*func) pointer
     %r8    : uc_link from ucontext structure
     %r9    : address of setcontext
     %r14   : return address to uc_link trampoline
     %r15   : stack pointer.

   The trampoline looks like this:
     basr  %r14,%r7
     lr    %r2,%r8
     br    %r9.  */

void
__makecontext (ucontext_t *ucp, void (*func) (void), int argc, ...)
{
  extern void __makecontext_ret (void);
  unsigned long int *sp;
  va_list ap;

  sp = (unsigned long int *) (((unsigned long int) ucp->uc_stack.ss_sp
			       + ucp->uc_stack.ss_size) & -8L);

  /* Set the return address to trampoline.  */
  ucp->uc_mcontext.gregs[14] = (long int) __makecontext_ret;

  /* Set register parameters.  */
  va_start (ap, argc);
  for (int i = 0; i < argc && i < 5; ++i)
    ucp->uc_mcontext.gregs[2 + i] = va_arg (ap, long int);

  /* The remaining arguments go to the overflow area.  */
  if (argc > 5)
    {
      sp -= argc - 5;
      for (int i = 5; i < argc; ++i)
	sp[i - 5] = va_arg (ap, long int);
    }
  va_end (ap);

  /* Make room for the save area and set the backchain.  */
  sp -= 24;
  *sp = 0;

  /* Pass (*func) to __makecontext_ret in %r7.  */
  ucp->uc_mcontext.gregs[7] = (long int) func;

  /* Pass ucp->uc_link to __makecontext_ret in %r8.  */
  ucp->uc_mcontext.gregs[8] = (long int) ucp->uc_link;

  /* Pass address of setcontext in %r9.  */
  ucp->uc_mcontext.gregs[9] = (long int) &setcontext;

  /* Set stack pointer.  */
  ucp->uc_mcontext.gregs[15] = (long int) sp;
}

weak_alias (__makecontext, makecontext)
