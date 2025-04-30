/* @(#)s_ldexp.c 5.1 93/09/24 */
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
static char rcsid[] = "$NetBSD: s_ldexp.c,v 1.6 1995/05/10 20:47:40 jtc Exp $";
#endif

#include <math.h>
#include <errno.h>

FLOAT
M_SUF (__ldexp) (FLOAT value, int exp)
{
	if(!isfinite(value)||value==0) return value + value;
	value = M_SCALBN(value,exp);
	if(!isfinite(value)||value==0) __set_errno (ERANGE);
	return value;
}

declare_mgen_alias (__ldexp, ldexp)
strong_alias (M_SUF (__ldexp), M_SUF (__wrap_scalbn))
declare_mgen_alias (__wrap_scalbn, scalbn)

/* Note, versioning issues are punted to ldbl-opt in this case.  */
