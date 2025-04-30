/* Create new context for ARC.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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
#include <sys/ucontext.h>

void
__makecontext (ucontext_t *ucp, void (*func) (void), int argc, ...)
{
  extern void __startcontext (void) attribute_hidden;
  unsigned long int sp, *r;
  va_list vl;
  int i, reg_args, stack_args;

  sp = ((unsigned long int) ucp->uc_stack.ss_sp + ucp->uc_stack.ss_size) & ~7;

  ucp->uc_mcontext.__sp = sp;
  ucp->uc_mcontext.__fp = 0;

  /* __startcontext is sort of trampoline to invoke @func
     From setcontext pov, the resume address is __startcontext,
     set it up in BLINK place holder.  */

  ucp->uc_mcontext.__blink = (unsigned long int) &__startcontext;

  /* __startcontext passed 2 types of args
       - args to @func setup in canonical r0-r7
       - @func and next function in r14,r15.   */

  ucp->uc_mcontext.__r14 = (unsigned long int) func;
  ucp->uc_mcontext.__r15 = (unsigned long int) ucp->uc_link;

  r = &ucp->uc_mcontext.__r0;

  va_start (vl, argc);

  reg_args = argc > 8 ? 8 : argc;
  for (i = 0; i < reg_args; i++)
    *r-- = va_arg (vl, unsigned long int);

  stack_args = argc - reg_args;

  if (__glibc_unlikely (stack_args > 0))
    {
      sp -= stack_args * sizeof (unsigned long int);
      ucp->uc_mcontext.__sp = sp;
      r = (unsigned long int *) sp;

      for (i = 0; i < stack_args; i++)
        *r++ = va_arg (vl, unsigned long int);
    }

  va_end (vl);
}

weak_alias (__makecontext, makecontext)
