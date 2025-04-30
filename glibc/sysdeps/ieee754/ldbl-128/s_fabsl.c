/* s_fabsl.c -- long double version of s_fabs.c.
 * Conversion to IEEE quad long double by Jakub Jelinek, jj@ultra.linux.cz.
 */

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

#if defined(LIBM_SCCS) && !defined(lint)
static char rcsid[] = "$NetBSD: $";
#endif

/*
 * fabsl(x) returns the absolute value of x.
 */

#include <math.h>
#include <math_private.h>
#include <libm-alias-ldouble.h>

_Float128 __fabsl(_Float128 x)
{
	uint64_t hx;
	GET_LDOUBLE_MSW64(hx,x);
	SET_LDOUBLE_MSW64(x,hx&0x7fffffffffffffffLL);
        return x;
}
libm_alias_ldouble (__fabs, fabs)
