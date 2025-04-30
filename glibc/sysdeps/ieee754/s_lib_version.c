/* @(#)s_lib_ver.c 5.1 93/09/24 */
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
static char rcsid[] = "$NetBSD: s_lib_version.c,v 1.6 1995/05/10 20:47:44 jtc Exp $";
#endif

/*
 * MACRO for standards
 */

#include <math-svid-compat.h>

/*
 * define and initialize _LIB_VERSION
 */
#undef _LIB_VERSION
#if LIBM_SVID_COMPAT
_LIB_VERSION_TYPE _LIB_VERSION_INTERNAL = _POSIX_;
strong_alias (_LIB_VERSION_INTERNAL, _LIB_VERSION);
compat_symbol (libm, _LIB_VERSION_INTERNAL, _LIB_VERSION, GLIBC_2_0);
#endif
