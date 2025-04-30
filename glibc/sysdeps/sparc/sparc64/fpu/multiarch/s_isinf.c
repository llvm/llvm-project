/* isinf ifunc resolver, Linux/sparc64 version.
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

#define __isinf __redirect___isinf
#define __isinff __redirect___isinff
#define __isinfl __redirect___isinfl
#include <math.h>
#undef __isinf
#undef __isinff
#undef __isinfl
#include <sparc-ifunc.h>

extern __typeof (isinf) __isinf_vis3 attribute_hidden;
extern __typeof (isinf) __isinf_generic attribute_hidden;

sparc_libm_ifunc_redirected (__redirect___isinf, __isinf,
			     hwcap & HWCAP_SPARC_VIS3
			     ? __isinf_vis3
			     : __isinf_generic)

sparc_ifunc_redirected_hidden_def (__redirect___isinf, __isinf)
weak_alias (__isinf, isinf)
