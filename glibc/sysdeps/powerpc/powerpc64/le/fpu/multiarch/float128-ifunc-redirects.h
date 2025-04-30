/* _Float128 redirects for ppc64le multiarch env.
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

#ifndef _FLOAT128_IFUNC_REDIRECTS
#define _FLOAT128_IFUNC_REDIRECTS 1

#include <float128-ifunc-macros.h>

F128_REDIR_PFX_R (sqrtf128, __,);
F128_REDIR_PFX_R (rintf128, __,);
F128_REDIR_PFX_R (ceilf128, __,);
F128_REDIR_PFX_R (floorf128, __,);
F128_REDIR_PFX_R (truncf128, __,);
F128_REDIR_PFX_R (roundf128, __,);
F128_REDIR_PFX_R (fabsf128, __,);

extern __typeof (ldexpf128) F128_SFX_APPEND (__ldexpf128);

#define __ldexpf128 F128_SFX_APPEND (__ldexpf128)

/* libm_hidden_proto is disabled by the time we reach here.
   Ensure some internally called functions are still called
   without going through the PLT.  Note, this code is only
   included when building libm.  */
hidden_proto (__fpclassifyf128)
hidden_proto (__issignalingf128)

#endif /* _FLOAT128_IFUNC_REDIRECTS */
