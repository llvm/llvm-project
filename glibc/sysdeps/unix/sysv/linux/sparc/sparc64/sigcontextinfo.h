/* Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jj@ultra.linux.cz>, 1999.

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

#ifndef _SIGCONTEXTINFO_H
#define _SIGCONTEXTINFO_H

#include <bits/types/siginfo_t.h>

/* The sparc64 kernel signal frame for SA_SIGINFO is defined as:

   struct rt_signal_frame
   {
     struct sparc_stackf ss;
     siginfo_t info;
     struct pt_regs regs;
     __siginfo_fpu_t *fpu_save;
     stack_t stack;
     sigset_t mask;
     __siginfo_rwin_t *rwin_save;
   };

   Unlike other architectures, sparc64 passe the siginfo_t INFO pointer
   as the third argument to a sa_sigaction handler with SA_SIGINFO enabled.  */

#ifndef STACK_BIAS
#define STACK_BIAS 2047
#endif

struct pt_regs
{
  unsigned long int u_regs[16];
  unsigned long int tstate;
  unsigned long int tpc;
  unsigned long int tnpc;
  unsigned int y;
  unsigned int magic;
};

static inline uintptr_t
sigcontext_get_pc (const siginfo_t *ctx)
{
  struct pt_regs *regs = (struct pt_regs*) ((siginfo_t *)(ctx) + 1);
  return regs->tpc;
}

#endif
