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
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

/* System V/i386 ABI compliant context switching support.  */

#ifndef _SYS_UCONTEXT_H
#define _SYS_UCONTEXT_H	1

#include <features.h>

#include <bits/types/sigset_t.h>
#include <bits/types/stack_t.h>


/* Type for general register.  */
typedef int greg_t;

/* Number of general registers.  */
#define __NGREG	19
#ifdef __USE_MISC
# define NGREG	__NGREG
#endif

/* Container for all general registers.  */
typedef greg_t gregset_t[__NGREG];

#ifdef __USE_MISC
/* Number of each register is the `gregset_t' array.  */
enum
{
  REG_GS = 0,
# define REG_GS	REG_GS
  REG_FS,
# define REG_FS	REG_FS
  REG_ES,
# define REG_ES	REG_ES
  REG_DS,
# define REG_DS	REG_DS
  REG_EDI,
# define REG_EDI	REG_EDI
  REG_ESI,
# define REG_ESI	REG_ESI
  REG_EBP,
# define REG_EBP	REG_EBP
  REG_ESP,
# define REG_ESP	REG_ESP
  REG_EBX,
# define REG_EBX	REG_EBX
  REG_EDX,
# define REG_EDX	REG_EDX
  REG_ECX,
# define REG_ECX	REG_ECX
  REG_EAX,
# define REG_EAX	REG_EAX
  REG_TRAPNO,
# define REG_TRAPNO	REG_TRAPNO
  REG_ERR,
# define REG_ERR	REG_ERR
  REG_EIP,
# define REG_EIP	REG_EIP
  REG_CS,
# define REG_CS	REG_CS
  REG_EFL,
# define REG_EFL	REG_EFL
  REG_UESP,
# define REG_UESP	REG_UESP
  REG_SS
# define REG_SS	REG_SS
};
#endif

#ifdef __USE_MISC
# define __ctx(fld) fld
# define __ctxt(tag) tag
#else
# define __ctx(fld) __ ## fld
# define __ctxt(tag) /* Empty.  */
#endif

/* Structure to describe FPU registers.  */
typedef struct
  {
    union
      {
	struct __ctxt(fpchip_state)
	  {
	    int __ctx(state)[27];
	    int __ctx(status);
	  } __ctx(fpchip_state);

	struct __ctxt(fp_emul_space)
	  {
	    char __ctx(fp_emul)[246];
	    char __ctx(fp_epad)[2];
	  } __ctx(fp_emul_space);

	int __ctx(f_fpregs)[62];
      } __ctx(fp_reg_set);

    long int __ctx(f_wregs)[33];
  } fpregset_t;

/* Context to describe whole processor state.  */
typedef struct
  {
    gregset_t __ctx(gregs);
    fpregset_t __ctx(fpregs);
  } mcontext_t;

/* Userlevel context.  */
typedef struct ucontext_t
  {
    unsigned long int __ctx(uc_flags);
    struct ucontext_t *uc_link;
    sigset_t uc_sigmask;
    stack_t uc_stack;
    mcontext_t uc_mcontext;
    long int __glibc_reserved1[5];
  } ucontext_t;

#undef __ctx
#undef __ctxt

#endif /* sys/ucontext.h */
