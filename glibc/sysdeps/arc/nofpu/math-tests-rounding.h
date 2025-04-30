/* Configuration for math tests: rounding mode support.  ARC version.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#ifndef ARC_NOFPU_MATH_TESTS_ROUNDING_H
#define ARC_NOFPU_MATH_TESTS_ROUNDING_H 1

/* Soft-float only supports to-nearest rounding mode.  */
#define ROUNDING_TESTS_float(MODE)		((MODE) == FE_TONEAREST)
#define ROUNDING_TESTS_double(MODE)		((MODE) == FE_TONEAREST)
#define ROUNDING_TESTS_long_double(MODE)	((MODE) == FE_TONEAREST)

#endif
