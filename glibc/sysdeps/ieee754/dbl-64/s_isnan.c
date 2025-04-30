/* @(#)s_isnan.c 5.1 93/09/24 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

/*
 * isnan(x) returns 1 is x is nan, else 0;
 * no branching!
 */

#include <math.h>
#include <math_private.h>
#include <ldbl-classify-compat.h>
#include <shlib-compat.h>
#include <stdint.h>

int
__isnan (double x)
{
  int64_t hx;
  EXTRACT_WORDS64 (hx, x);
  hx &= UINT64_C (0x7fffffffffffffff);
  hx = UINT64_C (0x7ff0000000000000) - hx;
  return (int)(((uint64_t)hx)>>63);
}
hidden_def (__isnan)
weak_alias (__isnan, isnan)
#ifdef NO_LONG_DOUBLE
# if LDBL_CLASSIFY_COMPAT && SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_23)
compat_symbol (libc, __isnan, __isnanl, GLIBC_2_0);
# endif
weak_alias (__isnan, isnanl)
#endif
