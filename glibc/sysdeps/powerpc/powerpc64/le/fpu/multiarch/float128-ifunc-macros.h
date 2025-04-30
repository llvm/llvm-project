/* _Float128 aliasing macro support for ifunc generation on PPC.
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

#ifndef _FLOAT128_IFUNC_MACROS_PPC64LE
#define _FLOAT128_IFUNC_MACROS_PPC64LE 1

/* Bring in the various alias-providing headers, and disable
   _Float128 related macros.  This prevents exporting any ABI
   from _Float128 implementation objects.  */
#include <libm-alias-float128.h>
#include <libm-alias-finite.h>

#undef libm_alias_float128_r
#undef libm_alias_finite
#undef libm_alias_exclusive_ldouble
#undef libm_alias_float128_other_r_ldbl
#undef declare_mgen_finite_alias
#undef declare_mgen_alias
#undef declare_mgen_alias_r

#define libm_alias_finite(from, to)
#define libm_alias_float128_r(from, to, r)
#define libm_alias_exclusive_ldouble(from, to)
#define libm_alias_float128_other_r_ldbl(from, to, r)
#define declare_mgen_finite_alias(from, to)
#define declare_mgen_alias(from, to)
#define declare_mgen_alias_r(from, to)

/*  Likewise, disable hidden symbol support.  This is not needed
    for the implementation objects as the redirects already give
    us this support.  This also means any non-_Float128 headers
    which provide hidden_def's should be included prior to this
    header (e.g fenv.h).  */
#undef libm_hidden_def
#define libm_hidden_def(func)
#undef libm_hidden_proto
#define libm_hidden_proto(f)

#include <float128-ifunc-redirect-macros.h>

#endif /* _FLOAT128_IFUNC_MACROS_PPC64LE */
