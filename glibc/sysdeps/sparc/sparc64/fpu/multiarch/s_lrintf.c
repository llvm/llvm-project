/* lrintf/llrintf ifunc resolver, Linux/sparc64 version.
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

#define lrintf __redirect_lrintf
#define llrintf __redirect_llrintf
#define __lrintf __redirect_lrintf
#define __llrintf __redirect_llrintf
#include <math.h>
#undef lrintf
#undef llrintf
#undef __lrintf
#undef __llrintf
#include <sparc-ifunc.h>
#include <libm-alias-float.h>

extern __typeof (__redirect_lrintf) __lrintf_vis3 attribute_hidden;
extern __typeof (__redirect_lrintf) __lrintf_generic attribute_hidden;

sparc_libm_ifunc_redirected (__redirect_lrintf, __lrintf,
			     hwcap & HWCAP_SPARC_VIS3
			     ? __lrintf_vis3
			     : __lrintf_generic);
libm_alias_float (__lrint, lrint)
strong_alias (__lrintf, __llrintf)
libm_alias_float (__llrint, llrint)
