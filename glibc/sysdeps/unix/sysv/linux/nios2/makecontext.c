/* Create new context.
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

#include <sysdep.h>
#include <stdarg.h>
#include <stdint.h>
#include <ucontext.h>

/* makecontext sets up a stack and the registers for the
   user context.  The stack looks like this:

               +-----------------------+
	       | padding as required   |
               +-----------------------+
    sp ->      | parameters 5 to n     |
               +-----------------------+

   The registers are set up like this:
     r4--r7 : parameter 1 to 4
     r16    : uc_link
     sp     : stack pointer.
*/

void
__makecontext (ucontext_t *ucp, void (*func) (void), int argc, ...)
{
  extern void __startcontext (void);
  unsigned long *sp;
  va_list ap;
  int i;

  sp = (unsigned long *)
    ((uintptr_t) ucp->uc_stack.ss_sp + ucp->uc_stack.ss_size);

  /* Allocate stack arguments.  */
  sp -= argc < 4 ? 0 : argc - 4;

  /* Keep the stack aligned.  */
  sp = (unsigned long*) (((uintptr_t) sp) & -4L);

  /* Init version field.  */
  ucp->uc_mcontext.version = 2;
  /* Keep uc_link in r16.  */
  ucp->uc_mcontext.regs[15] = (uintptr_t) ucp->uc_link;
  /* Return address points to __startcontext().  */
  ucp->uc_mcontext.regs[23] = (uintptr_t) &__startcontext;
  /* Frame pointer is null.  */
  ucp->uc_mcontext.regs[24] = (uintptr_t) 0;
  /* Restart in user-space starting at 'func'.  */
  ucp->uc_mcontext.regs[27] = (uintptr_t) func;
  /* Set stack pointer.  */
  ucp->uc_mcontext.regs[28] = (uintptr_t) sp;

  va_start (ap, argc);
  for (i = 0; i < argc; ++i)
    if (i < 4)
      ucp->uc_mcontext.regs[i + 3] = va_arg (ap, unsigned long);
    else
      sp[i - 4] = va_arg (ap, unsigned long);

  va_end (ap);
}

weak_alias (__makecontext, makecontext)
