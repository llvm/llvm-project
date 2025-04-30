/* signbit ifunc resolver, Linux/sparc64 version.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#define __signbit __redirect_signbit
#include <math.h>
#undef __signbit

#include <sparc-ifunc.h>

extern __typeof (__redirect_signbit) __signbit_vis3 attribute_hidden;
extern __typeof (__redirect_signbit) __signbit_generic attribute_hidden;

sparc_libm_ifunc_redirected (__redirect_signbit, __signbit,
			     hwcap & HWCAP_SPARC_VIS3
			     ? __signbit_vis3
			     : __signbit_generic);

/* On 64-bit the double version will also always work for
   long-double-precision since in both cases the word with the
   sign bit in it is passed always in register %f0.  */
strong_alias (__signbit, __signbitl)
sparc_ifunc_redirected_hidden_def (__redirect_signbit, __signbitl)
