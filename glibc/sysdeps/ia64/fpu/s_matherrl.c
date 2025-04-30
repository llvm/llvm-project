/* Derived from:  */
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


#include <math.h>
#include <math-svid-compat.h>
#include "libm_support.h"

#if LIBM_SVID_COMPAT
int
weak_function
__matherrl(struct exceptionl *x)
{
	int n=0;
	if(x->arg1!=x->arg1) return 0;
	return n;
}
compat_symbol (libm, __matherrl, matherrl, GLIBC_2_2_3);
#endif
