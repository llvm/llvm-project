/* Define __clog10l compat symbol for clog10 for ldbl-opt.
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

#include <math_ldbl_opt.h>
#include <first-versions.h>
#include <math-type-macros-double.h>

#include <s_clog10_template.c>

#if LONG_DOUBLE_COMPAT (libm, FIRST_VERSION_libm___clog10l)
strong_alias (__clog10, __clog10l_alias)
compat_symbol (libm, __clog10l_alias, __clog10l, FIRST_VERSION_libm___clog10l);
#endif
