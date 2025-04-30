/* Compute positive difference, sparc 32-bit.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#include <math_ldbl_opt.h>
#include <first-versions.h>
#include <sparc-ifunc.h>
#include <math.h>
#include <libm-alias-double.h>

extern double __fdim_vis3 (double, double);
extern double __fdim_generic (double, double);

sparc_libm_ifunc(__fdim, hwcap & HWCAP_SPARC_VIS3 ? __fdim_vis3 : __fdim_generic);
libm_alias_double (__fdim, fdim)
