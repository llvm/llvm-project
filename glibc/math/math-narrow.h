/* Helper macros for functions returning a narrower type.
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

#ifndef	_MATH_NARROW_H
#define	_MATH_NARROW_H	1

#include <bits/floatn.h>
#include <bits/long-double.h>
#include <errno.h>
#include <fenv.h>
#include <ieee754.h>
#include <math-barriers.h>
#include <math_private.h>
#include <fenv_private.h>

/* Carry out a computation using round-to-odd.  The computation is
   EXPR; the union type in which to store the result is UNION and the
   subfield of the "ieee" field of that union with the low part of the
   mantissa is MANTISSA; SUFFIX is the suffix for the libc_fe* macros
   to ensure that the correct rounding mode is used, for platforms
   with multiple rounding modes where those macros set only the
   relevant mode.  This macro does not work correctly if the sign of
   an exact zero result depends on the rounding mode, so that case
   must be checked for separately.  */
#define ROUND_TO_ODD(EXPR, UNION, SUFFIX, MANTISSA)			\
  ({									\
    fenv_t env;								\
    UNION u;								\
									\
    libc_feholdexcept_setround ## SUFFIX (&env, FE_TOWARDZERO);		\
    u.d = (EXPR);							\
    math_force_eval (u.d);						\
    u.ieee.MANTISSA							\
      |= libc_feupdateenv_test ## SUFFIX (&env, FE_INEXACT) != 0;	\
									\
    u.d;								\
  })

/* Check for error conditions from a narrowing add function returning
   RET with arguments X and Y and set errno as needed.  Overflow and
   underflow can occur for finite arguments and a domain error for
   infinite ones.  */
#define CHECK_NARROW_ADD(RET, X, Y)			\
  do							\
    {							\
      if (!isfinite (RET))				\
	{						\
	  if (isnan (RET))				\
	    {						\
	      if (!isnan (X) && !isnan (Y))		\
		__set_errno (EDOM);			\
	    }						\
	  else if (isfinite (X) && isfinite (Y))	\
	    __set_errno (ERANGE);			\
	}						\
      else if ((RET) == 0 && (X) != -(Y))		\
	__set_errno (ERANGE);				\
    }							\
  while (0)

/* Implement narrowing add using round-to-odd.  The arguments are X
   and Y, the return type is TYPE and UNION, MANTISSA and SUFFIX are
   as for ROUND_TO_ODD.  */
#define NARROW_ADD_ROUND_TO_ODD(X, Y, TYPE, UNION, SUFFIX, MANTISSA)	\
  do									\
    {									\
      TYPE ret;								\
									\
      /* Ensure a zero result is computed in the original rounding	\
	 mode.  */							\
      if ((X) == -(Y))							\
	ret = (TYPE) ((X) + (Y));					\
      else								\
	ret = (TYPE) ROUND_TO_ODD (math_opt_barrier (X) + (Y),		\
				   UNION, SUFFIX, MANTISSA);		\
									\
      CHECK_NARROW_ADD (ret, (X), (Y));					\
      return ret;							\
    }									\
  while (0)

/* Implement a narrowing add function that is not actually narrowing
   or where no attempt is made to be correctly rounding (the latter
   only applies to IBM long double).  The arguments are X and Y and
   the return type is TYPE.  */
#define NARROW_ADD_TRIVIAL(X, Y, TYPE)		\
  do						\
    {						\
      TYPE ret;					\
						\
      ret = (TYPE) ((X) + (Y));			\
      CHECK_NARROW_ADD (ret, (X), (Y));		\
      return ret;				\
    }						\
  while (0)

/* Check for error conditions from a narrowing subtract function
   returning RET with arguments X and Y and set errno as needed.
   Overflow and underflow can occur for finite arguments and a domain
   error for infinite ones.  */
#define CHECK_NARROW_SUB(RET, X, Y)			\
  do							\
    {							\
      if (!isfinite (RET))				\
	{						\
	  if (isnan (RET))				\
	    {						\
	      if (!isnan (X) && !isnan (Y))		\
		__set_errno (EDOM);			\
	    }						\
	  else if (isfinite (X) && isfinite (Y))	\
	    __set_errno (ERANGE);			\
	}						\
      else if ((RET) == 0 && (X) != (Y))		\
	__set_errno (ERANGE);				\
    }							\
  while (0)

/* Implement narrowing subtract using round-to-odd.  The arguments are
   X and Y, the return type is TYPE and UNION, MANTISSA and SUFFIX are
   as for ROUND_TO_ODD.  */
#define NARROW_SUB_ROUND_TO_ODD(X, Y, TYPE, UNION, SUFFIX, MANTISSA)	\
  do									\
    {									\
      TYPE ret;								\
									\
      /* Ensure a zero result is computed in the original rounding	\
	 mode.  */							\
      if ((X) == (Y))							\
	ret = (TYPE) ((X) - (Y));					\
      else								\
	ret = (TYPE) ROUND_TO_ODD (math_opt_barrier (X) - (Y),		\
				   UNION, SUFFIX, MANTISSA);		\
									\
      CHECK_NARROW_SUB (ret, (X), (Y));					\
      return ret;							\
    }									\
  while (0)

/* Implement a narrowing subtract function that is not actually
   narrowing or where no attempt is made to be correctly rounding (the
   latter only applies to IBM long double).  The arguments are X and Y
   and the return type is TYPE.  */
#define NARROW_SUB_TRIVIAL(X, Y, TYPE)		\
  do						\
    {						\
      TYPE ret;					\
						\
      ret = (TYPE) ((X) - (Y));			\
      CHECK_NARROW_SUB (ret, (X), (Y));		\
      return ret;				\
    }						\
  while (0)

/* Check for error conditions from a narrowing multiply function
   returning RET with arguments X and Y and set errno as needed.
   Overflow and underflow can occur for finite arguments and a domain
   error for Inf * 0.  */
#define CHECK_NARROW_MUL(RET, X, Y)			\
  do							\
    {							\
      if (!isfinite (RET))				\
	{						\
	  if (isnan (RET))				\
	    {						\
	      if (!isnan (X) && !isnan (Y))		\
		__set_errno (EDOM);			\
	    }						\
	  else if (isfinite (X) && isfinite (Y))	\
	    __set_errno (ERANGE);			\
	}						\
      else if ((RET) == 0 && (X) != 0 && (Y) != 0)	\
	__set_errno (ERANGE);				\
    }							\
  while (0)

/* Implement narrowing multiply using round-to-odd.  The arguments are
   X and Y, the return type is TYPE and UNION, MANTISSA and SUFFIX are
   as for ROUND_TO_ODD.  */
#define NARROW_MUL_ROUND_TO_ODD(X, Y, TYPE, UNION, SUFFIX, MANTISSA)	\
  do									\
    {									\
      TYPE ret;								\
									\
      ret = (TYPE) ROUND_TO_ODD (math_opt_barrier (X) * (Y),		\
				 UNION, SUFFIX, MANTISSA);		\
									\
      CHECK_NARROW_MUL (ret, (X), (Y));					\
      return ret;							\
    }									\
  while (0)

/* Implement a narrowing multiply function that is not actually
   narrowing or where no attempt is made to be correctly rounding (the
   latter only applies to IBM long double).  The arguments are X and Y
   and the return type is TYPE.  */
#define NARROW_MUL_TRIVIAL(X, Y, TYPE)		\
  do						\
    {						\
      TYPE ret;					\
						\
      ret = (TYPE) ((X) * (Y));			\
      CHECK_NARROW_MUL (ret, (X), (Y));		\
      return ret;				\
    }						\
  while (0)

/* Check for error conditions from a narrowing divide function
   returning RET with arguments X and Y and set errno as needed.
   Overflow, underflow and divide-by-zero can occur for finite
   arguments and a domain error for Inf / Inf and 0 / 0.  */
#define CHECK_NARROW_DIV(RET, X, Y)			\
  do							\
    {							\
      if (!isfinite (RET))				\
	{						\
	  if (isnan (RET))				\
	    {						\
	      if (!isnan (X) && !isnan (Y))		\
		__set_errno (EDOM);			\
	    }						\
	  else if (isfinite (X))			\
	    __set_errno (ERANGE);			\
	}						\
      else if ((RET) == 0 && (X) != 0 && !isinf (Y))	\
	__set_errno (ERANGE);				\
    }							\
  while (0)

/* Implement narrowing divide using round-to-odd.  The arguments are
   X and Y, the return type is TYPE and UNION, MANTISSA and SUFFIX are
   as for ROUND_TO_ODD.  */
#define NARROW_DIV_ROUND_TO_ODD(X, Y, TYPE, UNION, SUFFIX, MANTISSA)	\
  do									\
    {									\
      TYPE ret;								\
									\
      ret = (TYPE) ROUND_TO_ODD (math_opt_barrier (X) / (Y),		\
				 UNION, SUFFIX, MANTISSA);		\
									\
      CHECK_NARROW_DIV (ret, (X), (Y));					\
      return ret;							\
    }									\
  while (0)

/* Implement a narrowing divide function that is not actually
   narrowing or where no attempt is made to be correctly rounding (the
   latter only applies to IBM long double).  The arguments are X and Y
   and the return type is TYPE.  */
#define NARROW_DIV_TRIVIAL(X, Y, TYPE)		\
  do						\
    {						\
      TYPE ret;					\
						\
      ret = (TYPE) ((X) / (Y));			\
      CHECK_NARROW_DIV (ret, (X), (Y));		\
      return ret;				\
    }						\
  while (0)

/* The following macros declare aliases for a narrowing function.  The
   sole argument is the base name of a family of functions, such as
   "add".  If any platform changes long double format after the
   introduction of narrowing functions, in a way requiring symbol
   versioning compatibility, additional variants of these macros will
   be needed.  */

#define libm_alias_float_double_main(func)	\
  weak_alias (__f ## func, f ## func)		\
  weak_alias (__f ## func, f32 ## func ## f64)	\
  weak_alias (__f ## func, f32 ## func ## f32x)

#ifdef NO_LONG_DOUBLE
# define libm_alias_float_double(func)		\
  libm_alias_float_double_main (func)		\
  weak_alias (__f ## func, f ## func ## l)
#else
# define libm_alias_float_double(func)		\
  libm_alias_float_double_main (func)
#endif

#define libm_alias_float32x_float64_main(func)			\
  weak_alias (__f32x ## func ## f64, f32x ## func ## f64)

#ifdef NO_LONG_DOUBLE
# define libm_alias_float32x_float64(func)		\
  libm_alias_float32x_float64_main (func)		\
  weak_alias (__f32x ## func ## f64, d ## func ## l)
#elif defined __LONG_DOUBLE_MATH_OPTIONAL
# define libm_alias_float32x_float64(func)			\
  libm_alias_float32x_float64_main (func)			\
  weak_alias (__f32x ## func ## f64, __nldbl_d ## func ## l)
#else
# define libm_alias_float32x_float64(func)	\
  libm_alias_float32x_float64_main (func)
#endif

#if __HAVE_FLOAT128 && !__HAVE_DISTINCT_FLOAT128
# define libm_alias_float_ldouble_f128(func)		\
  weak_alias (__f ## func ## l, f32 ## func ## f128)
# define libm_alias_double_ldouble_f128(func)		\
  weak_alias (__d ## func ## l, f32x ## func ## f128)	\
  weak_alias (__d ## func ## l, f64 ## func ## f128)
#else
# define libm_alias_float_ldouble_f128(func)
# define libm_alias_double_ldouble_f128(func)
#endif

#if __HAVE_FLOAT64X_LONG_DOUBLE
# define libm_alias_float_ldouble_f64x(func)		\
  weak_alias (__f ## func ## l, f32 ## func ## f64x)
# define libm_alias_double_ldouble_f64x(func)		\
  weak_alias (__d ## func ## l, f32x ## func ## f64x)	\
  weak_alias (__d ## func ## l, f64 ## func ## f64x)
#else
# define libm_alias_float_ldouble_f64x(func)
# define libm_alias_double_ldouble_f64x(func)
#endif

#define libm_alias_float_ldouble(func)		\
  weak_alias (__f ## func ## l, f ## func ## l) \
  libm_alias_float_ldouble_f128 (func)		\
  libm_alias_float_ldouble_f64x (func)

#define libm_alias_double_ldouble(func)		\
  weak_alias (__d ## func ## l, d ## func ## l) \
  libm_alias_double_ldouble_f128 (func)		\
  libm_alias_double_ldouble_f64x (func)

#define libm_alias_float64x_float128(func)			\
  weak_alias (__f64x ## func ## f128, f64x ## func ## f128)

#define libm_alias_float32_float128_main(func)			\
  weak_alias (__f32 ## func ## f128, f32 ## func ## f128)

#define libm_alias_float64_float128_main(func)			\
  weak_alias (__f64 ## func ## f128, f64 ## func ## f128)	\
  weak_alias (__f64 ## func ## f128, f32x ## func ## f128)

#include <math-narrow-alias-float128.h>

#endif /* math-narrow.h.  */
