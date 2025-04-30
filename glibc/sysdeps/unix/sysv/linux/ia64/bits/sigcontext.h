/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jes Sorensen <jes@linuxcare.com>, July 2000

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

#ifndef _BITS_SIGCONTEXT_H
#define _BITS_SIGCONTEXT_H 1

#if !defined _SIGNAL_H && !defined _SYS_UCONTEXT_H
# error "Never use <bits/sigcontext.h> directly; include <signal.h> instead."
#endif

#define __need_size_t
#include <stddef.h>
#include <bits/sigstack.h>
#include <bits/types/struct_sigstack.h>
#include <bits/types/stack_t.h>
#include <bits/ss_flags.h>

struct __ia64_fpreg
  {
    union
      {
	unsigned long bits[2];
      } u;
  } __attribute__ ((__aligned__ (16)));

struct sigcontext
{
  unsigned long int sc_flags;	/* see manifest constants below */
  unsigned long int sc_nat;	/* bit i == 1 iff scratch reg gr[i] is a NaT */
  stack_t sc_stack;		/* previously active stack */

  unsigned long int sc_ip;	/* instruction pointer */
  unsigned long int sc_cfm;	/* current frame marker */
  unsigned long int sc_um;	/* user mask bits */
  unsigned long int sc_ar_rsc;	/* register stack configuration register */
  unsigned long int sc_ar_bsp;	/* backing store pointer */
  unsigned long int sc_ar_rnat;	/* RSE NaT collection register */
  unsigned long int sc_ar_ccv;	/* compare & exchange compare value register */
  unsigned long int sc_ar_unat;	/* ar.unat of interrupted context */
  unsigned long int sc_ar_fpsr;	/* floating-point status register */
  unsigned long int sc_ar_pfs;	/* previous function state */
  unsigned long int sc_ar_lc;	/* loop count register */
  unsigned long int sc_pr;	/* predicate registers */
  unsigned long int sc_br[8];	/* branch registers */
  unsigned long int sc_gr[32];	/* general registers (static partition) */
  struct __ia64_fpreg sc_fr[128];	/* floating-point registers */
  unsigned long int sc_rbs_base;/* NULL or new base of sighandler's rbs */
  unsigned long int sc_loadrs;	/* see description above */
  unsigned long int sc_ar25;	/* cmp8xchg16 uses this */
  unsigned long int sc_ar26;	/* rsvd for scratch use */
  unsigned long int sc_rsvd[12];/* reserved for future use */

  /* sc_mask is actually an sigset_t but we don't want to
   * include the kernel headers here. */
  unsigned long int sc_mask;	/* signal mask to restore after handler returns */
};

/* sc_flag bit definitions. */
#define IA64_SC_FLAG_ONSTACK_BIT	0	/* is handler running on signal stack? */
#define IA64_SC_FLAG_IN_SYSCALL_BIT	1	/* did signal interrupt a syscall? */
#define IA64_SC_FLAG_FPH_VALID_BIT	2	/* is state in f[32]-f[127] valid? */

#define IA64_SC_FLAG_ONSTACK		(1 << IA64_SC_FLAG_ONSTACK_BIT)
#define IA64_SC_FLAG_IN_SYSCALL		(1 << IA64_SC_FLAG_IN_SYSCALL_BIT)
#define IA64_SC_FLAG_FPH_VALID		(1 << IA64_SC_FLAG_FPH_VALID_BIT)

#endif /* _BITS_SIGCONTEXT_H */
