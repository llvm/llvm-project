/* __mpn_mul_1 ifunc resolver, Linux/sparc64 version.
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

#include <gmp.h>
#include <sparc-ifunc.h>

extern __typeof (mpn_mul_1) __mpn_mul_1_vis3 attribute_hidden;
extern __typeof (mpn_mul_1) __mpn_mul_1_generic attribute_hidden;

sparc_libm_ifunc (__mpn_mul_1,
		  hwcap & HWCAP_SPARC_VIS3
		  ? __mpn_mul_1_vis3
		  : __mpn_mul_1_generic)
