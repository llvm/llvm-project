/* Return 1 if argument is a NaN, else 0.
   Copyright (C) 2007-2021 Free Software Foundation, Inc.
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

/* Ugly kludge to avoid declarations.  */
#define __isnanf	not___isnanf
#define isnanf		not_isnanf
#define __GI___isnanf	not__GI___isnanf

#include <math.h>
#include <math_private.h>
#include <math_ldbl_opt.h>

#undef __isnanf
#undef isnanf
#undef __GI___isnanf

int
__isnan (double x)
{
  uint64_t ix;
  EXTRACT_WORDS64 (ix, x);
  return ix * 2 > 0xffe0000000000000ul;
}

hidden_def (__isnan)
weak_alias (__isnan, isnan)

/* It turns out that the 'double' version will also always work for
   single-precision.  */
strong_alias (__isnan, __isnanf)
weak_alias (__isnan, isnanf)

/* ??? GCC 4.8 fails to look through chains of aliases with asm names
   attached.  Work around this for now.  */
hidden_ver (__isnan, __isnanf)

#ifdef NO_LONG_DOUBLE
strong_alias (__isnan, __isnanl)
weak_alias (__isnan, isnanl)
#endif
#if LONG_DOUBLE_COMPAT(libc, GLIBC_2_0)
compat_symbol (libc, __isnan, __isnanl, GLIBC_2_0);
compat_symbol (libc, isnan, isnanl, GLIBC_2_0);
#endif
