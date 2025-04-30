/* Configuration for math tests: support for exceptions.  Generic version.
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

#ifndef _MATH_TESTS_EXCEPTIONS_H
#define _MATH_TESTS_EXCEPTIONS_H 1

/* Indicate whether to run tests of floating-point exceptions for a
   given floating-point type, given that the exception macros are
   defined.  All are run unless overridden.  */
#define EXCEPTION_TESTS_float	1
#define EXCEPTION_TESTS_double	1
#define EXCEPTION_TESTS_long_double	1
#define EXCEPTION_TESTS_float128	1

#endif /* math-tests-exceptions.h.  */
