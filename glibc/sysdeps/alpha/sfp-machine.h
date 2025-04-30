/* Machine-dependent software floating-point definitions.
   Alpha userland IEEE 128-bit version.
   Copyright (C) 2004-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <fenv_libc.h>

#define _FP_W_TYPE_SIZE		64
#define _FP_W_TYPE		unsigned long
#define _FP_WS_TYPE		signed long
#define _FP_I_TYPE		long

#define _FP_MUL_MEAT_S(R,X,Y)					\
  _FP_MUL_MEAT_1_imm(_FP_WFRACBITS_S,R,X,Y)
#define _FP_MUL_MEAT_D(R,X,Y)					\
  _FP_MUL_MEAT_1_wide(_FP_WFRACBITS_D,R,X,Y,umul_ppmm)
#define _FP_MUL_MEAT_Q(R,X,Y)					\
  _FP_MUL_MEAT_2_wide(_FP_WFRACBITS_Q,R,X,Y,umul_ppmm)

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

/* Alpha Architecture Handbook, 4.7.10.4 sez that we should prefer any
   type of NaN in Fb, then Fa.  */
#define _FP_CHOOSENAN(fs, wc, R, X, Y, OP)                      \
  do {                                                          \
    R##_s = Y##_s;                                              \
    _FP_FRAC_COPY_##wc(R,X);                                    \
    R##_c = FP_CLS_NAN;                                         \
  } while (0)

/* Rounding mode settings.  */
#define FP_RND_NEAREST		FE_TONEAREST
#define FP_RND_ZERO		FE_TOWARDZERO
#define FP_RND_PINF		FE_UPWARD
#define FP_RND_MINF		FE_DOWNWARD

/* Obtain the current rounding mode.  It's given as an argument to
   all the Ots functions, with 4 meaning "dynamic".  */
#define FP_ROUNDMODE		_round

/* Exception flags. */
#define FP_EX_INVALID		FE_INVALID
#define FP_EX_OVERFLOW		FE_OVERFLOW
#define FP_EX_UNDERFLOW		FE_UNDERFLOW
#define FP_EX_DIVZERO		FE_DIVBYZERO
#define FP_EX_INEXACT		FE_INEXACT

#define _FP_TININESS_AFTER_ROUNDING 1

#define FP_INIT_ROUNDMODE					\
do {								\
  if (__builtin_expect (_round == 4, 0))			\
    {								\
      unsigned long t;						\
      __asm__ __volatile__("excb; mf_fpcr %0" : "=f"(t));	\
      _round = (t >> FPCR_ROUND_SHIFT) & 3;			\
    }								\
} while (0)

/* We copy the libm function into libc for soft-fp.  */
extern int __feraiseexcept (int __excepts) attribute_hidden;

#define FP_HANDLE_EXCEPTIONS					\
do {								\
  if (__builtin_expect (_fex, 0))				\
    __feraiseexcept (_fex);					\
} while (0)

#define FP_TRAPPING_EXCEPTIONS					\
  ((__ieee_get_fp_control () & SWCR_ENABLE_MASK) << SWCR_ENABLE_SHIFT)
