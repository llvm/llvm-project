/* Configuration for math tests: support for exceptions.  ARM no-FPU version.
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

#ifndef ARM_NOFPU_MATH_TESTS_EXCEPTIONS_H
#define ARM_NOFPU_MATH_TESTS_EXCEPTIONS_H 1

/* On systems with VFP support, but where glibc is built for
   soft-float, the libgcc functions used in libc and libm do not
   support exceptions.  */
#define EXCEPTION_TESTS_float	0
#define EXCEPTION_TESTS_double	0
#define EXCEPTION_TESTS_long_double	0

#endif /* math-tests-exceptions.h.  */
