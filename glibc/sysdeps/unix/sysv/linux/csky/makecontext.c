/* Create new context.  C-SKY version.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <stdarg.h>
#include <ucontext.h>

/* Number of arguments that go in registers.  */
#define NREG_ARGS  4

/* Take a context previously prepared via getcontext() and set to
   call func() with the given int only args.  */
void
__makecontext (ucontext_t *ucp, void (*func) (void), int argc, ...)
{
  extern void __startcontext (void);
  unsigned long *funcstack;
  va_list vl;
  unsigned long *regptr;
  unsigned int reg;
  int misaligned;

  /* Start at the top of stack.  */
  funcstack = (unsigned long *) (ucp->uc_stack.ss_sp + ucp->uc_stack.ss_size);

  /* Ensure the stack stays eight byte aligned.  */
  misaligned = ((unsigned long) funcstack & 4) != 0;

  if ((argc > NREG_ARGS) && (argc & 1) != 0)
    misaligned = !misaligned;

  if (misaligned)
    funcstack -= 1;

  va_start (vl, argc);

  /* Reserve space for the on-stack arguments.  */
  if (argc > NREG_ARGS)
    funcstack -= (argc - NREG_ARGS);

  ucp->uc_mcontext.__gregs.__usp = (unsigned long) funcstack;
  ucp->uc_mcontext.__gregs.__pc = (unsigned long) func;

  /* Exit to startcontext() with the next context in R9.  */
  ucp->uc_mcontext.__gregs.__regs[5] = (unsigned long) ucp->uc_link;
  ucp->uc_mcontext.__gregs.__lr = (unsigned long) __startcontext;

  /* The first four arguments go into registers.  */
  regptr = &(ucp->uc_mcontext.__gregs.__a0);

  for (reg = 0; (reg < argc) && (reg < NREG_ARGS); reg++)
    *regptr++ = va_arg (vl, unsigned long);

  /* And the remainder on the stack.  */
  for (; reg < argc; reg++)
    *funcstack++ = va_arg (vl, unsigned long);

  va_end (vl);
}
weak_alias (__makecontext, makecontext)
