/* isnan ifunc resolver, Linux/sparc64 version.
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

#define __isnan __redirect___isnan
#define __isnanf __redirect___isnanf
#define __isnanl __redirect___isnanl
#include <math.h>
#undef __isnan
#undef __isnanf
#undef __isnanl
#include <sparc-ifunc.h>

extern __typeof (isnan) __isnan_vis3 attribute_hidden;
extern __typeof (isnan) __isnan_generic attribute_hidden;

sparc_libm_ifunc_redirected (__redirect___isnan, __isnan,
			     hwcap & HWCAP_SPARC_VIS3
			     ? __isnan_vis3
			     : __isnan_generic);

sparc_ifunc_redirected_hidden_def (__redirect___isnan, __isnan)
weak_alias (__isnan, isnan)
