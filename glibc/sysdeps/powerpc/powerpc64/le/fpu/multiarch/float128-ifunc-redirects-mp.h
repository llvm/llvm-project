/* _Float128 multiarch redirects shared with math_private.h
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

#ifndef _FLOAT128_IFUNC_REDIRECTS_MP_H
#define _FLOAT128_IFUNC_REDIRECTS_MP_H 1

#include <float128-ifunc-redirect-macros.h>

F128_REDIR (__ieee754_acosf128)
F128_REDIR (__ieee754_acoshf128)
F128_REDIR (__ieee754_asinf128)
F128_REDIR (__ieee754_atan2f128)
F128_REDIR (__ieee754_atanhf128)
F128_REDIR (__ieee754_coshf128)
F128_REDIR (__ieee754_expf128)
F128_REDIR (__ieee754_exp10f128)
F128_REDIR (__ieee754_exp2f128)
F128_REDIR (__ieee754_fmodf128)
F128_REDIR (__ieee754_gammaf128)
F128_REDIR_R (__ieee754_gammaf128, _r)
F128_REDIR (__ieee754_hypotf128)
F128_REDIR (__ieee754_j0f128)
F128_REDIR (__ieee754_j1f128)
F128_REDIR (__ieee754_jnf128)
F128_REDIR (__ieee754_lgammaf128)
F128_REDIR_R (__ieee754_lgammaf128, _r)
F128_REDIR (__ieee754_logf128)
F128_REDIR (__ieee754_log10f128)
F128_REDIR (__ieee754_log2f128)
F128_REDIR (__ieee754_powf128)
F128_REDIR (__ieee754_remainderf128)
F128_REDIR (__ieee754_sinhf128)
F128_REDIR (__ieee754_sqrtf128)
F128_REDIR (__ieee754_y0f128)
F128_REDIR (__ieee754_y1f128)
F128_REDIR (__ieee754_ynf128)
F128_REDIR (__ieee754_scalbf128)
F128_REDIR (__ieee754_ilogbf128)
F128_REDIR (__ieee754_rem_pio2f128)
F128_REDIR (__kernel_sinf128)
F128_REDIR (__kernel_cosf128)
F128_REDIR (__kernel_tanf128)
F128_REDIR (__kernel_sincosf128)
F128_REDIR (__kernel_rem_pio2f128)
F128_REDIR (__x2y2m1f128)
F128_REDIR (__gamma_productf128)
F128_REDIR (__lgamma_negf128)

#endif /*_FLOAT128_IFUNC_REDIRECTS_MP_H */
