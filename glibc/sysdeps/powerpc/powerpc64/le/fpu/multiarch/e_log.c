/* Multiple versions of IEEE 754 log.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

#include <math.h>
#include "init-arch.h"
#include <libm-alias-finite.h>

extern double __redirect_ieee754_log (double);

extern __typeof (log) __log_ppc64 attribute_hidden;
#ifdef USE_PPC64_MCPU_POWER10
extern __typeof (log) __log_power10 attribute_hidden;
#endif

libc_ifunc_redirected (__redirect_ieee754_log, __ieee754_log,
#ifdef USE_PPC64_MCPU_POWER10
	    (hwcap2 & PPC_FEATURE2_ARCH_3_1)
	    ? __log_power10 :
#endif
            __log_ppc64);

libm_alias_finite (__ieee754_log, __log)
