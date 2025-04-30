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
 * finite(x) returns 1 is x is finite, else 0;
 * no branching!
 */

#include <math.h>
#include <math_private.h>
#include <ldbl-classify-compat.h>
#include <shlib-compat.h>
#include <stdint.h>

int
__finite (double x)
{
  int64_t lx;
  EXTRACT_WORDS64 (lx,x);
  return (int)((uint64_t)((lx & INT64_C(0x7ff0000000000000))
			  - INT64_C (0x7ff0000000000000)) >> 63);
}
hidden_def (__finite)
weak_alias (__finite, finite)
#ifdef NO_LONG_DOUBLE
# if LDBL_CLASSIFY_COMPAT
#  if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_23)
compat_symbol (libc, __finite, __finitel, GLIBC_2_0);
#  endif
#  if SHLIB_COMPAT (libm, GLIBC_2_1, GLIBC_2_23)
compat_symbol (libm, __finite, __finitel, GLIBC_2_1);
#  endif
# endif
weak_alias (__finite, finitel)
#endif
