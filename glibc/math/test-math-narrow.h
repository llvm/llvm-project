/* Common definitions for libm tests for narrowing scalar functions.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#define FUNC_TEST(function)						\
  FUNC_TEST_CONCAT (FUNC_NARROW_PREFIX, function, FUNC_NARROW_SUFFIX)
#define FUNC_TEST_CONCAT(prefix, function, suffix)	\
  _FUNC_TEST_CONCAT (prefix, function, suffix)
#define _FUNC_TEST_CONCAT(prefix, function, suffix)	\
  prefix ## function ## suffix
#define TEST_MATHVEC 0
#define TEST_NARROW 1
