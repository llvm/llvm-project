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

/* System V/m68k ABI compliant context switching support.  */

#ifndef _SYS_UCONTEXT_H
#define _SYS_UCONTEXT_H	1

#include <features.h>

#include <bits/types/sigset_t.h>
#include <bits/types/stack_t.h>


/* Type for general register.  */
typedef int greg_t;

/* Number of general registers.  */
#define __NGREG	18
#ifdef __USE_MISC
# define NGREG	__NGREG
#endif

/* Container for all general registers.  */
typedef greg_t gregset_t[__NGREG];

#ifdef __USE_MISC
/* Number of each register is the `gregset_t' array.  */
enum
{
  R_D0 = 0,
# define R_D0	R_D0
  R_D1 = 1,
# define R_D1	R_D1
  R_D2 = 2,
# define R_D2	R_D2
  R_D3 = 3,
# define R_D3	R_D3
  R_D4 = 4,
# define R_D4	R_D4
  R_D5 = 5,
# define R_D5	R_D5
  R_D6 = 6,
# define R_D6	R_D6
  R_D7 = 7,
# define R_D7	R_D7
  R_A0 = 8,
# define R_A0	R_A0
  R_A1 = 9,
# define R_A1	R_A1
  R_A2 = 10,
# define R_A2	R_A2
  R_A3 = 11,
# define R_A3	R_A3
  R_A4 = 12,
# define R_A4	R_A4
  R_A5 = 13,
# define R_A5	R_A5
  R_A6 = 14,
# define R_A6	R_A6
  R_A7 = 15,
# define R_A7	R_A7
  R_SP = 15,
# define R_SP	R_SP
  R_PC = 16,
# define R_PC	R_PC
  R_PS = 17
# define R_PS	R_PS
};

/* Structure to describe FPU registers.  */
typedef struct
{
  int f_pcr;
  int f_psr;
  int f_fpiaddr;
  int f_fpregs[8][3];
} fpregset_t;
#endif

#ifdef __USE_MISC
# define __ctx(fld) fld
#else
# define __ctx(fld) __ ## fld
#endif

/* Context to describe whole processor state.  */
typedef struct
{
  int __ctx(version);
  gregset_t __ctx(gregs);
} mcontext_t;

#ifdef __USE_MISC
# define MCONTEXT_VERSION 1
#endif

/* Userlevel context.  */
typedef struct ucontext_t
{
  unsigned long int __ctx(uc_flags);
  struct ucontext_t *uc_link;
  sigset_t uc_sigmask;
  stack_t uc_stack;
  mcontext_t uc_mcontext;
  long int __glibc_reserved1[201];
} ucontext_t;

#undef __ctx

#endif /* sys/ucontext.h */
