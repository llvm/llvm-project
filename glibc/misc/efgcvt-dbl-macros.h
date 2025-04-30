/* Macros for the implementation of *cvt functions, double version.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
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

#include <float.h>

#define FLOAT_TYPE double
#define FUNC_PREFIX
#define FLOAT_FMT_FLAG
#define FLOAT_NAME_EXT
#define FLOAT_MIN_10_EXP DBL_MIN_10_EXP
/* Actually we have to write (DBL_DIG + log10 (DBL_MAX_10_EXP)) but we
   don't have log10 available in the preprocessor.  */
#define MAXDIG (NDIGIT_MAX + 3)
#define FCVT_MAXDIG (DBL_MAX_10_EXP + MAXDIG)
#if DBL_MANT_DIG == 53
# define NDIGIT_MAX 17
#elif DBL_MANT_DIG == 24
# define NDIGIT_MAX 9
#elif DBL_MANT_DIG == 56
# define NDIGIT_MAX 18
#else
/* See IEEE 854 5.6, table 2 for this formula.  Unfortunately we need a
   compile time constant here, so we cannot use it.  */
# error "NDIGIT_MAX must be precomputed"
# define NDIGIT_MAX (lrint (ceil (M_LN2 / M_LN10 * DBL_MANT_DIG + 1.0)))
#endif
#if DBL_MIN_10_EXP == -37
# define FLOAT_MIN_10_NORM	1.0e-37
#elif DBL_MIN_10_EXP == -307
# define FLOAT_MIN_10_NORM	1.0e-307
#elif DBL_MIN_10_EXP == -4931
# define FLOAT_MIN_10_NORM	1.0e-4931
#else
/* libc can't depend on libm.  */
# error "FLOAT_MIN_10_NORM must be precomputed"
# define FLOAT_MIN_10_NORM	exp10 (DBL_MIN_10_EXP)
#endif
