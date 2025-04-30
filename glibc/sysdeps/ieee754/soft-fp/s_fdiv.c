/* Divide double values, narrowing the result to float, using soft-fp.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#define f32divf64 __hide_f32divf64
#define f32divf32x __hide_f32divf32x
#define fdivl __hide_fdivl
#include <math.h>
#undef f32divf64
#undef f32divf32x
#undef fdivl

#include <math-narrow.h>
#include <libc-diag.h>

/* R_f[01] are not set in cases where they are not used in packing,
   but the compiler does not see that they are set in all cases where
   they are used, resulting in warnings that they may be used
   uninitialized.  The location of the warning differs in different
   versions of GCC, it may be where R is defined using a macro or it
   may be where the macro is defined.  This happens only with -O1.  */
DIAG_PUSH_NEEDS_COMMENT;
DIAG_IGNORE_NEEDS_COMMENT (8, "-Wmaybe-uninitialized");
#include <soft-fp.h>
#include <single.h>
#include <double.h>

float
__fdiv (double x, double y)
{
  FP_DECL_EX;
  FP_DECL_D (X);
  FP_DECL_D (Y);
  FP_DECL_D (R);
  FP_DECL_S (RN);
  float ret;

  FP_INIT_ROUNDMODE;
  FP_UNPACK_D (X, x);
  FP_UNPACK_D (Y, y);
  FP_DIV_D (R, X, Y);
#if _FP_W_TYPE_SIZE < _FP_FRACBITS_D
  FP_TRUNC_COOKED (S, D, 1, 2, RN, R);
#else
  FP_TRUNC_COOKED (S, D, 1, 1, RN, R);
#endif
  FP_PACK_S (ret, RN);
  FP_HANDLE_EXCEPTIONS;
  CHECK_NARROW_DIV (ret, x, y);
  return ret;
}
DIAG_POP_NEEDS_COMMENT;

libm_alias_float_double (div)
