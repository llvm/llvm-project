/* Copyright (C) 1997-2021 Free Software Foundation, Inc.
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

/* Don't rely on this, the interface is currently messed up and may need to
   be broken to be fixed.  */
#ifndef _SYS_UCONTEXT_H
#define _SYS_UCONTEXT_H	1

#include <features.h>

#include <bits/types/sigset_t.h>
#include <bits/types/stack_t.h>


#ifdef __USE_MISC
# define __ctx(fld) fld
#else
# define __ctx(fld) __ ## fld
#endif

#ifdef __USE_MISC
/* Type for general register.  */
typedef unsigned long int greg_t;

/* Number of general registers.  */
# define NGREG	80
# define NFPREG	32

/* Container for all general registers.  */
typedef struct gregset
  {
    greg_t g_regs[32];
    greg_t sr_regs[8];
    greg_t cr_regs[24];
    greg_t g_pad[16];
  } gregset_t;

/* Container for all FPU registers.  */
typedef struct
  {
    double fp_dregs[32];
  } fpregset_t;
#endif

/* Context to describe whole processor state.  */
typedef struct
  {
    unsigned long int __ctx(sc_flags);
    unsigned long int __ctx(sc_gr)[32];
    unsigned long long int __ctx(sc_fr)[32];
    unsigned long int __ctx(sc_iasq)[2];
    unsigned long int __ctx(sc_iaoq)[2];
    unsigned long int __ctx(sc_sar);
  } mcontext_t;

/* Userlevel context.  */
typedef struct ucontext_t
  {
    unsigned long int __ctx(uc_flags);
    struct ucontext_t *uc_link;
    stack_t uc_stack;
    mcontext_t uc_mcontext;
    sigset_t uc_sigmask;
  } ucontext_t;

#undef __ctx

#endif /* sys/ucontext.h */
