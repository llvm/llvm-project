/* Machine-dependent software floating-point definitions.
   Sparc64 userland (_Q_* and _Qp_*) version.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Richard Henderson (rth@cygnus.com),
		  Jakub Jelinek (jj@ultra.linux.cz) and
		  David S. Miller (davem@redhat.com).

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

#include <fpu_control.h>
#include <fenv.h>
#include <stdlib.h>

#define _FP_W_TYPE_SIZE		64
#define _FP_W_TYPE		unsigned long
#define _FP_WS_TYPE		signed long
#define _FP_I_TYPE		long

/* Helper macros for _FP_MUL_MEAT_2_120_240_double.  */
#define _FP_MUL_MEAT_SET_FE_TZ		   			\
do {								\
  static fpu_control_t _fetz = _FPU_RC_DOWN;			\
  _FPU_SETCW(_fetz);						\
} while (0)
#ifndef _FP_MUL_MEAT_RESET_FE
#define _FP_MUL_MEAT_RESET_FE _FPU_SETCW(_fcw)
#endif

#define _FP_MUL_MEAT_S(R,X,Y)					\
  _FP_MUL_MEAT_1_imm(_FP_WFRACBITS_S,R,X,Y)
#define _FP_MUL_MEAT_D(R,X,Y)					\
  _FP_MUL_MEAT_1_wide(_FP_WFRACBITS_D,R,X,Y,umul_ppmm)
#define _FP_MUL_MEAT_Q(R,X,Y)					\
  _FP_MUL_MEAT_2_120_240_double(_FP_WFRACBITS_Q,R,X,Y,		\
				_FP_MUL_MEAT_SET_FE_TZ,		\
				_FP_MUL_MEAT_RESET_FE)

#define _FP_DIV_MEAT_S(R,X,Y)	_FP_DIV_MEAT_1_imm(S,R,X,Y,_FP_DIV_HELP_imm)
#define _FP_DIV_MEAT_D(R,X,Y)	_FP_DIV_MEAT_1_udiv_norm(D,R,X,Y)
#define _FP_DIV_MEAT_Q(R,X,Y)	_FP_DIV_MEAT_2_udiv(Q,R,X,Y)

#define _FP_NANFRAC_S		((_FP_QNANBIT_S << 1) - 1)
#define _FP_NANFRAC_D		((_FP_QNANBIT_D << 1) - 1)
#define _FP_NANFRAC_Q		((_FP_QNANBIT_Q << 1) - 1), -1
#define _FP_NANSIGN_S		0
#define _FP_NANSIGN_D		0
#define _FP_NANSIGN_Q		0

#define _FP_KEEPNANFRACP 1
#define _FP_QNANNEGATEDP 0

/* If one NaN is signaling and the other is not,
 * we choose that one, otherwise we choose Y.
 */
#define _FP_CHOOSENAN(fs, wc, R, X, Y, OP)			\
  do {								\
    if ((_FP_FRAC_HIGH_RAW_##fs(Y) & _FP_QNANBIT_##fs)		\
	&& !(_FP_FRAC_HIGH_RAW_##fs(X) & _FP_QNANBIT_##fs))	\
      {								\
	R##_s = X##_s;						\
	_FP_FRAC_COPY_##wc(R,X);				\
      }								\
    else							\
      {								\
	R##_s = Y##_s;						\
	_FP_FRAC_COPY_##wc(R,Y);				\
      }								\
    R##_c = FP_CLS_NAN;						\
  } while (0)

/* Obtain the current rounding mode. */
#ifndef FP_ROUNDMODE
#define FP_ROUNDMODE	((_fcw >> 30) & 0x3)
#endif

/* Exception flags. */
#define FP_EX_INVALID		(1 << 4)
#define FP_EX_OVERFLOW		(1 << 3)
#define FP_EX_UNDERFLOW		(1 << 2)
#define FP_EX_DIVZERO		(1 << 1)
#define FP_EX_INEXACT		(1 << 0)

#define _FP_TININESS_AFTER_ROUNDING 0

#define _FP_DECL_EX \
  fpu_control_t _fcw __attribute__ ((unused)) = (FP_RND_NEAREST << 30)

#define FP_INIT_ROUNDMODE					\
do {								\
  _FPU_GETCW(_fcw);						\
} while (0)

#define FP_TRAPPING_EXCEPTIONS ((_fcw >> 23) & 0x1f)
#define FP_INHIBIT_RESULTS ((_fcw >> 23) & _fex)

/* Simulate exceptions using double arithmetics. */
extern void __Qp_handle_exceptions(int exc);

#define FP_HANDLE_EXCEPTIONS					\
do {								\
  if (!_fex)							\
    {								\
      /* This is the common case, so we do it inline.		\
       * We need to clear cexc bits if any.			\
       */							\
      __asm__ __volatile__("fzero %%f62\n\t"			\
			   "faddd %%f62, %%f62, %%f62"		\
			   : : : "f62");			\
    }								\
  else								\
    {								\
      __Qp_handle_exceptions (_fex);				\
    }								\
} while (0)

#define QP_HANDLE_EXCEPTIONS(_a)				\
do {								\
  if ((_fcw >> 23) & _fex)					\
    {								\
      _a;							\
    }								\
  else								\
    {								\
      _fcw = (_fcw & ~0x1fL) | (_fex << 5) | _fex;		\
      _FPU_SETCW(_fcw);						\
    }								\
} while (0)

#define QP_NO_EXCEPTIONS					\
  __asm ("fzero %%f62\n\t"					\
	 "faddd %%f62, %%f62, %%f62" : : : "f62")

#define QP_CLOBBER "memory", "f52", "f54", "f56", "f58", "f60", "f62"
#define QP_CLOBBER_CC QP_CLOBBER , "cc"
