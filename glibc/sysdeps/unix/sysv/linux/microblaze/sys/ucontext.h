/* Data structures for user-level context switching.  MicroBlaze version.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
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

typedef struct
  {
    struct
      {
	unsigned long int __ctx(r0);
	unsigned long int __ctx(r1);
	unsigned long int __ctx(r2);
	unsigned long int __ctx(r3);
	unsigned long int __ctx(r4);
	unsigned long int __ctx(r5);
	unsigned long int __ctx(r6);
	unsigned long int __ctx(r7);
	unsigned long int __ctx(r8);
	unsigned long int __ctx(r9);
	unsigned long int __ctx(r10);
	unsigned long int __ctx(r11);
	unsigned long int __ctx(r12);
	unsigned long int __ctx(r13);
	unsigned long int __ctx(r14);
	unsigned long int __ctx(r15);
	unsigned long int __ctx(r16);
	unsigned long int __ctx(r17);
	unsigned long int __ctx(r18);
	unsigned long int __ctx(r19);
	unsigned long int __ctx(r20);
	unsigned long int __ctx(r21);
	unsigned long int __ctx(r22);
	unsigned long int __ctx(r23);
	unsigned long int __ctx(r24);
	unsigned long int __ctx(r25);
	unsigned long int __ctx(r26);
	unsigned long int __ctx(r27);
	unsigned long int __ctx(r28);
	unsigned long int __ctx(r29);
	unsigned long int __ctx(r30);
	unsigned long int __ctx(r31);
	unsigned long int __ctx(pc);
	unsigned long int __ctx(msr);
	unsigned long int __ctx(ear);
	unsigned long int __ctx(esr);
	unsigned long int __ctx(fsr);
	int __ctx(pt_mode);
      } __ctx(regs);
    unsigned long int __ctx(oldmask);
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
