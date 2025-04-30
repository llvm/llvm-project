/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by David Huggins-Daines <dhd@debian.org>

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

#ifndef _FENV_H
# error "Never use <bits/fenv.h> directly; include <fenv.h> instead."
#endif

/* Define bits representing the exception.  We use the values of the
   appropriate enable bits in the FPU status word (which,
   coincidentally, are the same as the flag bits, but shifted right by
   27 bits).  */
enum
{
  FE_INVALID =
#define FE_INVALID	(1<<4) /* V */
    FE_INVALID,
  FE_DIVBYZERO =
#define FE_DIVBYZERO	(1<<3) /* Z */
    FE_DIVBYZERO,
  FE_OVERFLOW =
#define FE_OVERFLOW	(1<<2) /* O */
    FE_OVERFLOW,
  FE_UNDERFLOW =
#define FE_UNDERFLOW	(1<<1) /* U */
    FE_UNDERFLOW,
  FE_INEXACT =
#define FE_INEXACT	(1<<0) /* I */
    FE_INEXACT,
};

#define FE_ALL_EXCEPT \
	(FE_INEXACT | FE_DIVBYZERO | FE_UNDERFLOW | FE_OVERFLOW | FE_INVALID)

/* The PA-RISC FPU supports all of the four defined rounding modes.
   We use the values of the RM field in the floating point status
   register for the appropriate macros.  */
enum
  {
    FE_TONEAREST =
#define FE_TONEAREST	(0 << 9)
      FE_TONEAREST,
    FE_TOWARDZERO =
#define FE_TOWARDZERO	(1 << 9)
      FE_TOWARDZERO,
    FE_UPWARD =
#define FE_UPWARD	(2 << 9)
      FE_UPWARD,
    FE_DOWNWARD =
#define FE_DOWNWARD	(3 << 9)
      FE_DOWNWARD,
  };

/* Type representing exception flags. */
typedef unsigned int fexcept_t;

/* Type representing floating-point environment.  This structure
   corresponds to the layout of the status and exception words in the
   register file. The exception registers are never saved/stored by
   userspace. This structure is also not correctly aligned ever, in
   an ABI error we left out __aligned(8) and subsequently all of our
   fenv functions must accept unaligned input, align the input, and
   then use assembly to store fr0. This is a performance hit, but
   means the ABI is stable. */
typedef struct
{
  unsigned int __status_word;
  unsigned int __exception[7];
} fenv_t;

/* If the default argument is used we use this value.  */
#define FE_DFL_ENV ((const fenv_t *) -1)

#ifdef __USE_GNU
/* Floating-point environment where none of the exceptions are masked.  */
# define FE_NOMASK_ENV	((const fenv_t *) -2)
#endif

#if __GLIBC_USE (IEC_60559_BFP_EXT_C2X)
/* Type representing floating-point control modes.  */
typedef unsigned int femode_t;

/* Default floating-point control modes.  */
# define FE_DFL_MODE	((const femode_t *) -1L)
#endif
