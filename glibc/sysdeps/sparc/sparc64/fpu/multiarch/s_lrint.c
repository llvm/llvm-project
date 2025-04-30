/* lrint/llrint ifunc resolver, Linux/sparc64 version.
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

#define lrint __redirect_lrint
#define llrint __redirect_llrint
#define __lrint __redirect___lrint
#define __llrint __redirect___llrint
#include <math.h>
#undef lrint
#undef llrint
#undef __lrint
#undef __llrint
#include <sparc-ifunc.h>
#include <libm-alias-double.h>

extern __typeof (__redirect_lrint) __lrint_vis3 attribute_hidden;
extern __typeof (__redirect_lrint) __lrint_generic attribute_hidden;

sparc_libm_ifunc_redirected (__redirect_lrint, __lrint,
			     hwcap & HWCAP_SPARC_VIS3
			     ? __lrint_vis3
			     : __lrint_generic);
libm_alias_double (__lrint, lrint)
strong_alias (__lrint, __llrint)
libm_alias_double (__llrint, llrint)
