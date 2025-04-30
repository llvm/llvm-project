/* Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 1999.

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

/* The sparc32 kernel signal frame for SA_SIGINFO is defined as:

  struct rt_signal_frame32
  {
    struct sparc_stackf32 ss;
    compat_siginfo_t info;
    struct pt_regs32 regs;          <- void *ctx
    compat_sigset_t mask;
    u32 fpu_save;
    unsigned int insns[2];
    compat_stack_t stack;
    unsigned int extra_size;
    siginfo_extra_v8plus_t v8plus;
    u32 rwin_save;
  } __attribute__((aligned(8)));

  Unlike other architectures, sparc32 passes pt_regs32 REGS pointer as
  the third argument to a sa_sigaction handler with SA_SIGINFO enabled.  */

struct pt_regs32
{
  unsigned int psr;
  unsigned int pc;
  unsigned int npc;
  unsigned int y;
  unsigned int u_regs[16];
};

static inline uintptr_t
sigcontext_get_pc (const struct pt_regs32 *ctx)
{
  return ctx->pc;
}

#endif
