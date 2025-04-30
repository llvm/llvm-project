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

/* System V/mips ABI compliant context switching support.  */

#ifndef _SYS_UCONTEXT_H
#define _SYS_UCONTEXT_H	1

#include <features.h>

#include <bits/types.h>
#include <bits/types/sigset_t.h>
#include <bits/types/stack_t.h>

#include <sgidefs.h>


/* Type for general register.  */
#if _MIPS_SIM == _ABIO32
typedef __uint32_t greg_t;
#else
typedef __uint64_t greg_t;
#endif

/* Number of general registers.  */
#define __NGREG	36
#ifdef __USE_MISC
# define NGREG	__NGREG
#endif

/* Container for all general registers.  */
typedef greg_t gregset_t[__NGREG];

#ifdef __USE_MISC
/* Number of each register is the `gregset_t' array.  */
enum
{
  CTX_R0 = 0,
# define CTX_R0	CTX_R0
  CTX_AT = 1,
# define CTX_AT	CTX_AT
  CTX_V0 = 2,
# define CTX_V0	CTX_V0
  CTX_V1 = 3,
# define CTX_V1	CTX_V1
  CTX_A0 = 4,
# define CTX_A0	CTX_A0
  CTX_A1 = 5,
# define CTX_A1	CTX_A1
  CTX_A2 = 6,
# define CTX_A2	CTX_A2
  CTX_A3 = 7,
# define CTX_A3	CTX_A3
  CTX_T0 = 8,
# define CTX_T0	CTX_T0
  CTX_T1 = 9,
# define CTX_T1	CTX_T1
  CTX_T2 = 10,
# define CTX_T2	CTX_T2
  CTX_T3 = 11,
# define CTX_T3	CTX_T3
  CTX_T4 = 12,
# define CTX_T4	CTX_T4
  CTX_T5 = 13,
# define CTX_T5	CTX_T5
  CTX_T6 = 14,
# define CTX_T6	CTX_T6
  CTX_T7 = 15,
# define CTX_T7	CTX_T7
  CTX_S0 = 16,
# define CTX_S0	CTX_S0
  CTX_S1 = 17,
# define CTX_S1	CTX_S1
  CTX_S2 = 18,
# define CTX_S2	CTX_S2
  CTX_S3 = 19,
# define CTX_S3	CTX_S3
  CTX_S4 = 20,
# define CTX_S4	CTX_S4
  CTX_S5 = 21,
# define CTX_S5	CTX_S5
  CTX_S6 = 22,
# define CTX_S6	CTX_S6
  CTX_S7 = 23,
# define CTX_S7	CTX_S7
  CTX_T8 = 24,
# define CTX_T8	CTX_T8
  CTX_T9 = 25,
# define CTX_T9	CTX_T9
  CTX_K0 = 26,
# define CTX_K0	CTX_K0
  CTX_K1 = 27,
# define CTX_K1	CTX_K1
  CTX_GP = 28,
# define CTX_GP	CTX_GP
  CTX_SP = 29,
# define CTX_SP	CTX_SP
  CTX_S8 = 30,
# define CTX_S8	CTX_S8
  CTX_RA = 31,
# define CTX_RA	CTX_RA
  CTX_MDLO = 32,
# define CTX_MDLO	CTX_MDLO
  CTX_MDHI = 33,
# define CTX_MDHI	CTX_MDHI
  CTX_CAUSE = 34,
# define CTX_CAUSE	CTX_CAUSE
  CTX_EPC = 35,
# define CTX_EPC	CTX_EPC
};
#endif

#ifdef __USE_MISC
# define __ctx(fld) fld
#else
# define __ctx(fld) __ ## fld
#endif

/* Structure to describe FPU registers.  */
typedef struct
{
  union
  {
#if _MIPS_SIM == _ABIO32
    double __ctx(fp_dregs)[16];
    float __ctx(fp_fregs)[32];
    unsigned int __ctx(fp_regs)[32];
#else
    double __ctx(fp_dregs)[32];
    /* float __ctx(fp_fregs)[32]; */
    __uint64_t __ctx(fp_regs)[32];
#endif
  } __ctx(fp_r);
  unsigned int __ctx(fp_csr);
  unsigned int __ctx(fp_pad);
} fpregset_t;

/* Context to describe whole processor state.  */
typedef struct
{
  gregset_t __ctx(gpregs);
  fpregset_t __ctx(fpregs);
} mcontext_t;

/* Userlevel context.  */
typedef struct ucontext_t
{
#if _MIPS_SIM == _ABIO32
  unsigned long int __ctx(uc_flags);
#else
  __uint64_t __ctx(uc_flags);
#endif
  struct ucontext_t *uc_link;
  sigset_t uc_sigmask;
  stack_t uc_stack;
  mcontext_t uc_mcontext;
  int __glibc_reserved1[48];
} ucontext_t;

#undef __ctx

#endif /* sys/ucontext.h */
