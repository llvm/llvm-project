/* @(#)s_matherr.c 5.1 93/09/24 */
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
static char rcsid[] = "$NetBSD: s_matherr.c,v 1.6 1995/05/10 20:47:53 jtc Exp $";
#endif

#include <math-svid-compat.h>

#undef matherr
#if LIBM_SVID_COMPAT
int
weak_function
__matherr(struct exception *x)
{
	int n=0;
	if(x->arg1!=x->arg1) return 0;
	return n;
}
strong_alias (__matherr, matherr);
compat_symbol (libm, __matherr, matherr, GLIBC_2_0);
#endif
