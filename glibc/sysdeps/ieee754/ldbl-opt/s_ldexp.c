/* ldexp alias overrides for platforms where long double
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

#include <math-type-macros-double.h>
#include <s_ldexp_template.c>

#if IS_IN (libc) && LONG_DOUBLE_COMPAT (libc, GLIBC_2_0)
compat_symbol (libc, __ldexp, ldexpl, GLIBC_2_0);
compat_symbol (libc, __wrap_scalbn, scalbnl, GLIBC_2_0);
#endif
