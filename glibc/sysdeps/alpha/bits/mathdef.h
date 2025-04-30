/* Copyright (C) 1997-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef _COMPLEX_H
# error "Never use <bits/mathdef.h> directly; include <complex.h> instead"
#endif

#if defined _COMPLEX_H && !defined _COMPLEX_H_MATHDEF
# define _COMPLEX_H_MATHDEF 1
# if defined(__GNUC__) && !__GNUC_PREREQ(3,4)

/* Due to an ABI change, we need to remap the complex float symbols.  */
#  define _Mdouble_		float
#  define __MATHCALL(function, args) \
    __MATHDECL (_Complex float, function, args)
#  define __MATHDECL(type, function, args) \
    __MATHDECL_1(type, function##f, args, __c1_##function##f); \
    __MATHDECL_1(type, __##function##f, args, __c1_##function##f)
#  define __MATHDECL_1(type, function, args, alias) \
    extern type function args __asm__(#alias) __THROW

#  include <bits/cmathcalls.h>

#  undef _Mdouble_
#  undef __MATHCALL
#  undef __MATHDECL
#  undef __MATHDECL_1

# endif /* GNUC before 3.4 */
#endif /* COMPLEX_H */
