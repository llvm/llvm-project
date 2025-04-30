/* ABI compatibility for lgamma functions.  ia64 version.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#ifndef IA64_LGAMMA_COMPAT_H
#define IA64_LGAMMA_COMPAT_H 1

#include <math/lgamma-compat.h>

#undef LGFUNC
#if USE_AS_COMPAT
# define LGFUNC(FUNC) __ ## FUNC ## _compat
#else
# define LGFUNC(FUNC) __ieee754_ ## FUNC
#endif

#undef CALL_LGAMMA
#define CALL_LGAMMA(TYPE, FUNC, ARG)				\
  ({								\
    TYPE lgamma_tmp;						\
    extern int __signgam, signgam;				\
    lgamma_tmp = FUNC ((ARG), &__signgam, sizeof (__signgam));	\
    if (USE_AS_COMPAT)						\
      signgam = __signgam;					\
    lgamma_tmp;							\
  })

#endif /* lgamma-compat.h.  */
