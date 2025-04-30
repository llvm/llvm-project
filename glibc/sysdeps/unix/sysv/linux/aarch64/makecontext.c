/* Create new context.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.

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
    sp ->      | parameter 7-n         |
               +-----------------------+

   The registers are set up like this:
     %x0 .. %x7: parameter 1 to 8
     %x19   : uc_link
     %sp    : stack pointer.
*/

void
__makecontext (ucontext_t *ucp, void (*func) (void), int argc, ...)
{
  extern void __startcontext (void);
  uint64_t *sp;
  va_list ap;
  int i;

  sp = (uint64_t *)
    ((uintptr_t) ucp->uc_stack.ss_sp + ucp->uc_stack.ss_size);

  /* Allocate stack arguments.  */
  sp -= argc < 8 ? 0 : argc - 8;

  /* Keep the stack aligned.  */
  sp = (uint64_t *) (((uintptr_t) sp) & -16L);

  ucp->uc_mcontext.regs[19] = (uintptr_t) ucp->uc_link;
  ucp->uc_mcontext.sp = (uintptr_t) sp;
  ucp->uc_mcontext.pc = (uintptr_t) func;
  ucp->uc_mcontext.regs[29] = (uintptr_t) 0;
  ucp->uc_mcontext.regs[30] = (uintptr_t) &__startcontext;

  va_start (ap, argc);
  for (i = 0; i < argc; ++i)
    if (i < 8)
      ucp->uc_mcontext.regs[i] = va_arg (ap, uint64_t);
    else
      sp[i - 8] = va_arg (ap, uint64_t);

  va_end (ap);
}

weak_alias (__makecontext, makecontext)
