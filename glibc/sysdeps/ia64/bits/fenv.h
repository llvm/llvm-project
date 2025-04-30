/* Copyright (C) 1999-2021 Free Software Foundation, Inc.
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

#ifndef _FENV_H
# error "Never use <bits/fenv.h> directly; include <fenv.h> instead."
#endif


/* Define bits representing the exception.  We use the bit positions of
   the appropriate bits in the FPSR...  (Tahoe EAS 2.4 5-4)*/

enum
  {
    FE_INEXACT =
#define FE_INEXACT	(1 << 5)
      FE_INEXACT,

    FE_UNDERFLOW =
#define FE_UNDERFLOW	(1 << 4)
      FE_UNDERFLOW,

    FE_OVERFLOW =
#define FE_OVERFLOW	(1 << 3)
      FE_OVERFLOW,

    FE_DIVBYZERO =
#define FE_DIVBYZERO	(1 << 2)
      FE_DIVBYZERO,

    FE_UNNORMAL =
#define FE_UNNORMAL	(1 << 1)
      FE_UNNORMAL,

    FE_INVALID =
#define FE_INVALID	(1 << 0)
      FE_INVALID,

    FE_ALL_EXCEPT =
#define FE_ALL_EXCEPT	(FE_INEXACT | FE_UNDERFLOW | FE_OVERFLOW | FE_DIVBYZERO | FE_UNNORMAL | FE_INVALID)
      FE_ALL_EXCEPT
  };


enum
  {
    FE_TOWARDZERO =
#define FE_TOWARDZERO	3
      FE_TOWARDZERO,

    FE_UPWARD =
#define FE_UPWARD	2
      FE_UPWARD,

    FE_DOWNWARD =
#define FE_DOWNWARD	1
      FE_DOWNWARD,

    FE_TONEAREST =
#define FE_TONEAREST	0
      FE_TONEAREST,
  };


/* Type representing exception flags.  */
typedef unsigned long int fexcept_t;

/* Type representing floating-point environment.  */
typedef unsigned long int fenv_t;

/* If the default argument is used we use this value.  */
#define FE_DFL_ENV	((const fenv_t *) 0xc009804c0270033fUL)

#ifdef __USE_GNU
/* Floating-point environment where only FE_UNNORMAL is masked since this
   exception is not generally supported by glibc.  */
# define FE_NOMASK_ENV	((const fenv_t *) 0xc009804c02700302UL)

/* Floating-point environment with (processor-dependent) non-IEEE
   floating point.  In this case, turning on flush-to-zero mode for
   s0, s2, and s3.  */
# define FE_NONIEEE_ENV ((const fenv_t *) 0xc009a04d0270037fUL)
#endif

#if __GLIBC_USE (IEC_60559_BFP_EXT_C2X)
/* Type representing floating-point control modes.  */
typedef unsigned long int femode_t;

/* Default floating-point control modes.  */
# define FE_DFL_MODE	((const femode_t *) 0xc009804c0270033fUL)
#endif
