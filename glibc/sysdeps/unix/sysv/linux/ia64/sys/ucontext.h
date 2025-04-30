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

/*
 * These are here mostly for backwards compatibility with older Unices.
 * IA-64 Linux does not distinguish between "mcontext_t" and
 * "ucontext_t" as all the necessary info is inside the former.
 */

struct __ia64_fpreg_mcontext
  {
    union
      {
	unsigned long __ctx(bits)[2];
      } __ctx(u);
  } __attribute__ ((__aligned__ (16)));

typedef struct
  {
    unsigned long int __ctx(sc_flags);
    unsigned long int __ctx(sc_nat);
    stack_t __ctx(sc_stack);
    unsigned long int __ctx(sc_ip);
    unsigned long int __ctx(sc_cfm);
    unsigned long int __ctx(sc_um);
    unsigned long int __ctx(sc_ar_rsc);
    unsigned long int __ctx(sc_ar_bsp);
    unsigned long int __ctx(sc_ar_rnat);
    unsigned long int __ctx(sc_ar_ccv);
    unsigned long int __ctx(sc_ar_unat);
    unsigned long int __ctx(sc_ar_fpsr);
    unsigned long int __ctx(sc_ar_pfs);
    unsigned long int __ctx(sc_ar_lc);
    unsigned long int __ctx(sc_pr);
    unsigned long int __ctx(sc_br)[8];
    unsigned long int __ctx(sc_gr)[32];
    struct __ia64_fpreg_mcontext __ctx(sc_fr)[128];
    unsigned long int __ctx(sc_rbs_base);
    unsigned long int __ctx(sc_loadrs);
    unsigned long int __ctx(sc_ar25);
    unsigned long int __ctx(sc_ar26);
    unsigned long int __ctx(sc_rsvd)[12];
    unsigned long int __ctx(sc_mask);
  } mcontext_t;

#if __GNUC_PREREQ (3, 5)
# define _SC_GR0_OFFSET	\
	__builtin_offsetof (mcontext_t, __ctx(sc_gr)[0])
#elif defined __GNUC__
# define _SC_GR0_OFFSET	\
	(((char *) &((mcontext_t *) 0)->__ctx(sc_gr)[0]) - (char *) 0)
#else
# define _SC_GR0_OFFSET	0xc8	/* pray that this is correct... */
#endif

typedef struct ucontext_t
  {
    union
      {
	mcontext_t _mc;
	struct
	  {
	    unsigned long _pad[_SC_GR0_OFFSET/8];
	    struct ucontext_t *_link;	/* this should overlay sc_gr[0] */
	  }
	_uc;
      }
    _u;
  }
ucontext_t;

#define uc_mcontext	_u._mc
#define uc_sigmask	_u._mc.__ctx(sc_mask)
#define uc_stack	_u._mc.__ctx(sc_stack)
#define uc_link		_u._uc._link

#endif /* sys/ucontext.h */
