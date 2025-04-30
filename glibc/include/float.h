#ifndef _LIBC_FLOAT_H
#define _LIBC_FLOAT_H

#define __GLIBC_INTERNAL_STARTING_HEADER_IMPLEMENTATION
#include <bits/libc-header-start.h>

/* Some tests also define this macro, requiring a check here to avoid
   errors for duplicate definitions (see GCC bug 91451).  */
#if !defined _ISOMAC && !defined __STDC_WANT_IEC_60559_TYPES_EXT__
# define __STDC_WANT_IEC_60559_TYPES_EXT__
#endif

#include_next <float.h>

/* Supplement float.h macros for _FloatN and _FloatNx for older
   compilers which do not yet support the type.  These are described
   in TS 18661-3.  */
#include <features.h>
#include <bits/floatn.h>
#if !__GNUC_PREREQ (7, 0) && __GLIBC_USE (IEC_60559_TYPES_EXT)

# if __HAVE_FLOAT128
#  define FLT128_MANT_DIG	113
#  define FLT128_DECIMAL_DIG	36
#  define FLT128_DIG		33
#  define FLT128_MIN_EXP	(-16381)
#  define FLT128_MIN_10_EXP	(-4931)
#  define FLT128_MAX_EXP	16384
#  define FLT128_MAX_10_EXP	4932
#  define FLT128_MAX					\
  __f128 (1.18973149535723176508575932662800702e+4932)
#  define FLT128_EPSILON				\
  __f128 (1.92592994438723585305597794258492732e-34)
#  define FLT128_MIN					\
  __f128 (3.36210314311209350626267781732175260e-4932)
#  define FLT128_TRUE_MIN				\
  __f128 (6.47517511943802511092443895822764655e-4966)
# endif

/* Types other than _Float128 are typedefs for other types with old
   compilers.  */

# if __HAVE_FLOAT32
#  define FLT32_MANT_DIG	FLT_MANT_DIG
#  define FLT32_DECIMAL_DIG	FLT_DECIMAL_DIG
#  define FLT32_DIG		FLT_DIG
#  define FLT32_MIN_EXP		FLT_MIN_EXP
#  define FLT32_MIN_10_EXP	FLT_MIN_10_EXP
#  define FLT32_MAX_EXP		FLT_MAX_EXP
#  define FLT32_MAX_10_EXP	FLT_MAX_10_EXP
#  define FLT32_MAX		FLT_MAX
#  define FLT32_EPSILON		FLT_EPSILON
#  define FLT32_MIN		FLT_MIN
#  define FLT32_TRUE_MIN	FLT_TRUE_MIN
# endif

# if __HAVE_FLOAT64
#  define FLT64_MANT_DIG	DBL_MANT_DIG
#  define FLT64_DECIMAL_DIG	DBL_DECIMAL_DIG
#  define FLT64_DIG		DBL_DIG
#  define FLT64_MIN_EXP		DBL_MIN_EXP
#  define FLT64_MIN_10_EXP	DBL_MIN_10_EXP
#  define FLT64_MAX_EXP		DBL_MAX_EXP
#  define FLT64_MAX_10_EXP	DBL_MAX_10_EXP
#  define FLT64_MAX		DBL_MAX
#  define FLT64_EPSILON		DBL_EPSILON
#  define FLT64_MIN		DBL_MIN
#  define FLT64_TRUE_MIN	DBL_TRUE_MIN
# endif

# if __HAVE_FLOAT32X
#  define FLT32X_MANT_DIG	DBL_MANT_DIG
#  define FLT32X_DECIMAL_DIG	DBL_DECIMAL_DIG
#  define FLT32X_DIG		DBL_DIG
#  define FLT32X_MIN_EXP	DBL_MIN_EXP
#  define FLT32X_MIN_10_EXP	DBL_MIN_10_EXP
#  define FLT32X_MAX_EXP	DBL_MAX_EXP
#  define FLT32X_MAX_10_EXP	DBL_MAX_10_EXP
#  define FLT32X_MAX		DBL_MAX
#  define FLT32X_EPSILON	DBL_EPSILON
#  define FLT32X_MIN		DBL_MIN
#  define FLT32X_TRUE_MIN	DBL_TRUE_MIN
# endif

# if __HAVE_FLOAT64X
#  if __HAVE_FLOAT64X_LONG_DOUBLE
#   define FLT64X_MANT_DIG	LDBL_MANT_DIG
#   define FLT64X_DECIMAL_DIG	LDBL_DECIMAL_DIG
#   define FLT64X_DIG		LDBL_DIG
#   define FLT64X_MIN_EXP	LDBL_MIN_EXP
#   define FLT64X_MIN_10_EXP	LDBL_MIN_10_EXP
#   define FLT64X_MAX_EXP	LDBL_MAX_EXP
#   define FLT64X_MAX_10_EXP	LDBL_MAX_10_EXP
#   define FLT64X_MAX		LDBL_MAX
#   define FLT64X_EPSILON	LDBL_EPSILON
#   define FLT64X_MIN		LDBL_MIN
#   define FLT64X_TRUE_MIN	LDBL_TRUE_MIN
#  else
#   define FLT64X_MANT_DIG	FLT128_MANT_DIG
#   define FLT64X_DECIMAL_DIG	FLT128_DECIMAL_DIG
#   define FLT64X_DIG		FLT128_DIG
#   define FLT64X_MIN_EXP	FLT128_MIN_EXP
#   define FLT64X_MIN_10_EXP	FLT128_MIN_10_EXP
#   define FLT64X_MAX_EXP	FLT128_MAX_EXP
#   define FLT64X_MAX_10_EXP	FLT128_MAX_10_EXP
#   define FLT64X_MAX		FLT128_MAX
#   define FLT64X_EPSILON	FLT128_EPSILON
#   define FLT64X_MIN		FLT128_MIN
#   define FLT64X_TRUE_MIN	FLT128_TRUE_MIN
#  endif
# endif

#endif

#endif /* _LIBC_FLOAT_H */
