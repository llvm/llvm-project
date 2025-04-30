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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

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

/* Type for general register.  */
typedef long int greg_t;

/* Number of general registers.  */
#define __NGREG	33
#ifdef __USE_MISC
# define NGREG	__NGREG
#endif

/* Container for all general registers.  */
typedef greg_t gregset_t[__NGREG];

/* Type for floating-point register.  */
typedef long int fpreg_t;

/* Number of general registers.  */
#define __NFPREG	32
#ifdef __USE_MISC
# define NFPREG	__NFPREG
#endif

/* Container for all general registers.  */
typedef fpreg_t fpregset_t[__NFPREG];


/* A machine context is exactly a sigcontext.  */
typedef struct
  {
    long int __ctx(sc_onstack);
    long int __ctx(sc_mask);
    long int __ctx(sc_pc);
    long int __ctx(sc_ps);
    long int __ctx(sc_regs)[32];
    long int __ctx(sc_ownedfp);
    long int __ctx(sc_fpregs)[32];
    unsigned long int __ctx(sc_fpcr);
    unsigned long int __ctx(sc_fp_control);
    unsigned long int __glibc_reserved1, __glibc_reserved2;
    unsigned long int __ctx(sc_ssize);
    char *__ctx(sc_sbase);
    unsigned long int __ctx(sc_traparg_a0);
    unsigned long int __ctx(sc_traparg_a1);
    unsigned long int __ctx(sc_traparg_a2);
    unsigned long int __ctx(sc_fp_trap_pc);
    unsigned long int __ctx(sc_fp_trigger_sum);
    unsigned long int __ctx(sc_fp_trigger_inst);
  } mcontext_t;

/* Userlevel context.  */
typedef struct ucontext_t
  {
    unsigned long int __ctx(uc_flags);
    struct ucontext_t *uc_link;
    unsigned long __uc_osf_sigmask;
    stack_t uc_stack;
    mcontext_t uc_mcontext;
    sigset_t uc_sigmask;
  } ucontext_t;

#undef __ctx

#endif /* sys/ucontext.h */
