/* Configuration for math tests: sNaN support.  ia64 version.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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

#ifndef IA64_MATH_TESTS_SNAN_H
#define IA64_MATH_TESTS_SNAN_H 1

#define SNAN_TESTS_float	1
#define SNAN_TESTS_double	1
#define SNAN_TESTS_long_double	1

/* Before GCC 7, there is no built-in function to provide a __float128
   sNaN, so most sNaN tests for this type cannot work.  */
#if __GNUC_PREREQ (7, 0)
# define SNAN_TESTS_float128	1
#else
# define SNAN_TESTS_float128	0
#endif

#endif /* math-tests-snan.h.  */
