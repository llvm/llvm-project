/* finite ifunc resolver, Linux/sparc64 version.
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

#define __finite __redirect___finite
#define __finitef __redirect___finitef
#define __finitel __redirect___finitel
#include <math.h>
#undef __finite
#undef __finitef
#undef __finitel
#include <sparc-ifunc.h>

extern __typeof (finite) __finite_vis3 attribute_hidden;
extern __typeof (finite) __finite_generic attribute_hidden;

sparc_libm_ifunc_redirected (__redirect___finite, __finite,
			     hwcap & HWCAP_SPARC_VIS3
			     ? __finite_vis3
			     : __finite_generic);

sparc_ifunc_redirected_hidden_def (__redirect___finite, __finite)
weak_alias (__finite, finite)
