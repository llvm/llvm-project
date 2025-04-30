/* Create new context.
   Copyright (C) 2008-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by David S. Miller <davem@davemloft.net>, 2008.

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

#include <sysdep.h>
#include <stdarg.h>
#include <stdint.h>
#include <ucontext.h>

/* Sets up the outgoing arguments and the program counter for a user
   context for the requested function call.

   Returning to the correct parent context is pretty simple on
   Sparc.  We only need to link up the register windows correctly.
   Since global registers are clobbered by calls, we need not be
   concerned about those, and thus is all could be worked out without
   using a trampoline.

   Except that we must deal with the signal mask, thus a trampoline
   is unavoidable. 32-bit stackframe layout:
	      +-----------------------------------------+
	      | 7th and further parameters		|
	      +-----------------------------------------+
	      | backup storage for initial 6 parameters |
	      +-----------------------------------------+
	      | struct return pointer			|
	      +-----------------------------------------+
	      | 8 incoming registers			|
	      +-----------------------------------------+
	      | 8 local registers			|
     %sp -->  +-----------------------------------------+

*/

void
__makecontext (ucontext_t *ucp, void (*func) (void), int argc, ...)
{
  extern void __start_context (void);
  unsigned long int *sp;
  va_list ap;
  int i;

  sp = (unsigned long int *) (ucp->uc_stack.ss_sp + ucp->uc_stack.ss_size);
  sp -= 16 + 7 + argc;
  sp = (unsigned long int *) (((uintptr_t) sp) & ~(8 - 1));

  for (i = 0; i < 8; i++)
    sp[i + 8] = ucp->uc_mcontext.gregs[REG_O0 + i];

  /* The struct return pointer is essentially unused, so we can
     place the link there.  */
  sp[16] = (unsigned long int) ucp->uc_link;

  va_start (ap, argc);

  /* Fill in outgoing arguments, including those which will
     end up being passed on the stack.  */
  for (i = 0; i < argc; i++)
    {
      unsigned long int arg = va_arg (ap, unsigned long int);
      if (i < 6)
	ucp->uc_mcontext.gregs[REG_O0 + i] = arg;
      else
	sp[i + 23 - 6] = arg;
    }

  va_end (ap);

  ucp->uc_mcontext.gregs[REG_O6] = (unsigned long int) sp;

  ucp->uc_mcontext.gregs[REG_O7] = ((unsigned long int) __start_context) - 8;

  ucp->uc_mcontext.gregs[REG_PC] = (unsigned long int) func;
  ucp->uc_mcontext.gregs[REG_nPC] = ucp->uc_mcontext.gregs[REG_PC] + 4;
}

weak_alias (__makecontext, makecontext)
