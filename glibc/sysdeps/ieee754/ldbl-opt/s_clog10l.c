/* clog10l alias overrides for platforms where long double
   was previously not unique.
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

#define M_DECL_FUNC(x) __clog10l_internal
#include <math-type-macros-ldouble.h>

#undef declare_mgen_alias
#define declare_mgen_alias(from, to)

#include <s_clog10_template.c>

/* __clog10l is also a public symbol.  */
strong_alias (__clog10l_internal, __clog10_internal_l)
long_double_symbol (libm, __clog10l_internal, __clog10l);
long_double_symbol (libm, __clog10_internal_l, clog10l);
libm_alias_ldouble_other (__clog10_internal_, clog10)
