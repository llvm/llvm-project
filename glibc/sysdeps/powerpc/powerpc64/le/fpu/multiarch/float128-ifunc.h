/* _Float128 ifunc ABI/ifunc generation macros.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#ifndef _FLOAT128_IFUNC_H
#define _FLOAT128_IFUNC_H 1

/* Disable hidden prototypes.  These perform function renames which
   prevent the ifunc functions from working.  */
#undef hidden_proto
#define hidden_proto(x)
#define NO_MATH_REDIRECT 1

/* Include the real math.h to avoid optimizations caused by include/math.h
   (e.x fabsf128 prototype is masked by an inline definition).*/
#include <math/math.h>
#include <math_private.h>
#include <complex.h>
#include <first-versions.h>
#include <shlib-compat.h>
#include "init-arch.h"

#include <libm-alias-float128.h>
#include <libm-alias-finite.h>

/* _F128_IFUNC2(func, from, r)
      Generate an ifunc symbol func ## r from the symbols
	from ## {power8, power9} ## r

      We use the PPC hwcap bit HAS_IEEE128 to select between the two with
      the assumption all P9 features are available on such targets.  */
#define _F128_IFUNC2(func, from, r) \
	libc_ifunc (func ## r, (hwcap2 & PPC_FEATURE2_HAS_IEEE128) \
                                ? from ## _power9 ## r : from ## _power8 ## r)

/* _F128_IFUNC(func, r)
      Similar to above, except the exported symbol name trivially remaps from
      func ## {cpu} ## r to func ## r.  */
#define _F128_IFUNC(func, r) _F128_IFUNC2(func, func, r)

/* MAKE_IMPL_IFUNC2(func, pfx1, pfx2, r)
     Declare external symbols of type pfx1 ## func ## f128 ## r with the name
                                      pfx2 ## func ## f128 ## _{cpu} ## r
     which are exported as implementation specific symbols (i.e backing support
     for type classification macros).  */
#define MAKE_IMPL_IFUNC2(func, pfx1, pfx2, r) \
	extern __typeof (pfx1 ## func ## f128 ## r) pfx2 ## func ## f128_power8 ## r; \
	extern __typeof (pfx1 ## func ## f128 ## r) pfx2 ## func ## f128_power9 ## r; \
        _F128_IFUNC2 (__ ## func ## f128, pfx2 ## func ## f128, r);

/* GEN_COMPAT_R_e(f)
     Generate a compatability symbol for finite alias of ieee function.  */
#define GEN_COMPAT_R_e(f, r) \
	libm_alias_finite (__ieee754_ ## f ## f128 ## r, __ ## f ## f128 ## r)

#define GEN_COMPAT_e_acos(f) GEN_COMPAT_R_e(f,)
#define GEN_COMPAT_e_acosh(f) GEN_COMPAT_R_e(f,)
#define GEN_COMPAT_e_asin(f) GEN_COMPAT_R_e(f,)
#define GEN_COMPAT_e_sinh(f) GEN_COMPAT_R_e(f,)
#define GEN_COMPAT_e_atan2(f) GEN_COMPAT_R_e(f,)
#define GEN_COMPAT_e_atanh(f) GEN_COMPAT_R_e(f,)
#define GEN_COMPAT_e_cosh(f) GEN_COMPAT_R_e(f,)
#define GEN_COMPAT_e_exp10(f) GEN_COMPAT_R_e(f,)
#define GEN_COMPAT_e_exp2(f) GEN_COMPAT_R_e(f,)
#define GEN_COMPAT_e_exp(f) GEN_COMPAT_R_e(f,)
#define GEN_COMPAT_e_fmod(f) GEN_COMPAT_R_e(f,)
#define GEN_COMPAT_e_gamma_r(f) GEN_COMPAT_R_e(f,_r)
#define GEN_COMPAT_e_hypot(f) GEN_COMPAT_R_e(f,)
#define GEN_COMPAT_e_j0(f) GEN_COMPAT_R_e(f,) GEN_COMPAT_R_e(y0,)
#define GEN_COMPAT_e_j1(f) GEN_COMPAT_R_e(f,) GEN_COMPAT_R_e(y1,)
#define GEN_COMPAT_e_jn(f) GEN_COMPAT_R_e(f,) GEN_COMPAT_R_e(yn,)
#define GEN_COMPAT_e_lgamma_r(f) GEN_COMPAT_R_e(f,_r)
#define GEN_COMPAT_e_log10(f) GEN_COMPAT_R_e(f,)
#define GEN_COMPAT_e_log2(f) GEN_COMPAT_R_e(f,)
#define GEN_COMPAT_e_log(f) GEN_COMPAT_R_e(f,)
#define GEN_COMPAT_e_pow(f) GEN_COMPAT_R_e(f,)
#define GEN_COMPAT_e_remainder(f) GEN_COMPAT_R_e(f,)
#define GEN_COMPAT_e_sqrt(f) GEN_COMPAT_R_e(f,)

/* MAKE_IEEE_IFUNC_R(func, pfx, r)
    Declare an ieee ifunc symbol used internally by libm.  E.g __ieee754_sinf128  */
#define MAKE_IEEE_IFUNC_R(func, r) \
	extern __typeof (__ieee754_ ## func ## f128 ## r) __ieee754_ ## func ## f128_power8 ## r; \
	extern __typeof (__ieee754_ ## func ## f128 ## r) __ieee754_ ## func ## f128_power9 ## r; \
        _F128_IFUNC2 (__ieee754_ ## func ## f128, __ieee754_ ## func ## f128, r);

/* MAKE_IFUNCP_WRAP_R(w, func, r)
      Export a function which the implementation wraps with prefix w to
      to func ## r.  */
#define MAKE_IFUNCP_WRAP_R(w, func, r) \
	extern __typeof (func ## f128 ## r) __ ## func ## f128 ## r; \
	MAKE_IMPL_IFUNC2 (func,__,__ ## w, r) \
	weak_alias (__ ## func ## f128 ## r, func ## f128 ## r); \
	libm_alias_float128_other_r (__ ## func, func, r);

/* MAKE_IFUNCP_R(func, r)
    The default IFUNC generator for all libm _Float128 ABI except
    when specifically overwritten.  This is a convenience wrapper
    around MAKE_IFUNCP_R where w is not used.  */
#define MAKE_IFUNCP_R(func,r) MAKE_IFUNCP_WRAP_R (,func,r)

/* Generic aliasing functions.  */
#define DECL_ALIAS(f) MAKE_IFUNCP_R (f,)
#define DECL_ALIAS_s(f) MAKE_IFUNCP_R (f,)
#define DECL_ALIAS_w(f) MAKE_IFUNCP_R (f,)
#define DECL_ALIAS_e(f) MAKE_IEEE_IFUNC_R (f,)
#define DECL_ALIAS_k(f)
#define DECL_ALIAS_R_w(f) MAKE_IFUNCP_R (f, _r)
#define DECL_ALIAS_R_e(f) MAKE_IEEE_IFUNC_R (f,_r)

/* No symbols are defined in these helper/wrapper objects. */
#define DECL_ALIAS_lgamma_neg(x)
#define DECL_ALIAS_lgamma_product(x)
#define DECL_ALIAS_gamma_product(x)
#define DECL_ALIAS_x2y2m1(x)
#define DECL_ALIAS_s_log1p(x)
#define DECL_ALIAS_s_scalbln(x)
#define DECL_ALIAS_s_scalbn(x)

/* Ensure the wrapper functions get exposed via IFUNC, not the
   wrappee (e.g __w_log1pf128_power8 instead of __log1pf128_power8.  */
#define DECL_ALIAS_w_log1p(x) MAKE_IFUNCP_WRAP_R(w_,x,)
#define DECL_ALIAS_w_scalbln(x) MAKE_IFUNCP_WRAP_R(w_,x,)

/* These are declared in their respective jX objects.  */
#define DECL_ALIAS_w_j0(f) MAKE_IFUNCP_R (f,) MAKE_IFUNCP_R (y0,)
#define DECL_ALIAS_w_j1(f) MAKE_IFUNCP_R (f,) MAKE_IFUNCP_R (y1,)
#define DECL_ALIAS_w_jn(f) MAKE_IFUNCP_R (f,) MAKE_IFUNCP_R (yn,)
#define DECL_ALIAS_e_j0(f) MAKE_IEEE_IFUNC_R (f,) MAKE_IEEE_IFUNC_R (y0,)
#define DECL_ALIAS_e_j1(f) MAKE_IEEE_IFUNC_R (f,) MAKE_IEEE_IFUNC_R (y1,)
#define DECL_ALIAS_e_jn(f) MAKE_IEEE_IFUNC_R (f,) MAKE_IEEE_IFUNC_R (yn,)

#define DECL_ALIAS_s_erf(f) MAKE_IFUNCP_R (f,) MAKE_IFUNCP_R (erfc,)

/* scalbnf128 is an alias of ldexpf128.  */
#define DECL_ALIAS_s_ldexp(f) MAKE_IFUNCP_R (f,) MAKE_IFUNCP_WRAP_R (wrap_, scalbn,)

/* Declare an IFUNC for a symbol which only exists
   to provide long double == ieee128 ABI.  */
#define DECL_LDOUBLE_ALIAS(func, RTYPE, ARGS) \
	extern RTYPE func ARGS; \
	extern __typeof (func) func ## _power8; \
	extern __typeof (func) func ## _power9; \
	_F128_IFUNC ( func,)

/* Handle the special case functions which exist only to support
   ldouble == ieee128.  */
#define DECL_ALIAS_w_scalb(x) \
	DECL_LDOUBLE_ALIAS (__scalbf128,_Float128, (_Float128, _Float128)) \
	libm_alias_float128_other_r_ldbl (__scalb, scalb,)

#endif /* ifndef _FLOAT128_IFUNC_H  */
