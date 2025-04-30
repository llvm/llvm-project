/* Copyright (C) 1998-2021 Free Software Foundation, Inc.
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

#ifndef _SIGCONTEXTINFO_H
#define _SIGCONTEXTINFO_H

#include <signal.h>

static inline uintptr_t
sigcontext_get_pc (const ucontext_t *ctx)
{
#ifdef __powerpc64__
  return ctx->uc_mcontext.gp_regs[PT_NIP];
#else
  return ctx->uc_mcontext.uc_regs->gregs[PT_NIP];
#endif
}

#endif
