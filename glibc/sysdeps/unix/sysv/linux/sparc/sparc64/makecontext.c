/* Copyright (C) 2001-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>.

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

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <ucontext.h>

extern void __start_context (ucontext_t *ucp);

void
__makecontext (ucontext_t *ucp, void (*func) (void), int argc, ...)
{
  extern void __makecontext_ret (void);
  unsigned long *sp, *topsp;
  va_list ap;
  int i;

  sp = (unsigned long *) ((long) ucp->uc_stack.ss_sp + ucp->uc_stack.ss_size);
  sp -= (argc > 6 ? argc : 6) + 32;
  sp = (unsigned long *) (((long) sp) & -16L);
  topsp = sp + (argc > 6 ? argc : 6) + 16;

  ucp->uc_mcontext.mc_gregs[MC_PC] = (long) func;
  ucp->uc_mcontext.mc_gregs[MC_NPC] = ((long) func) + 4;
  ucp->uc_mcontext.mc_gregs[MC_O6] = ((long) sp) - 0x7ff;
  ucp->uc_mcontext.mc_gregs[MC_O7] = ((long) __start_context) - 8;
  ucp->uc_mcontext.mc_fp = ((long) topsp) - 0x7ff;
  ucp->uc_mcontext.mc_i7 = 0;
  topsp[14] = 0;
  topsp[15] = 0;
  sp[8] = (long) ucp->uc_link;
  va_start (ap, argc);
  for (i = 0; i < argc; ++i)
    if (i < 6)
      ucp->uc_mcontext.mc_gregs[MC_O0 + i] = va_arg (ap, long);
    else
      sp[16 + i] = va_arg (ap, long);
  va_end (ap);
}

weak_alias (__makecontext, makecontext)
