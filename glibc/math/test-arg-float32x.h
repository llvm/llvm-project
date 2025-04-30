/* Common definitions for libm tests for _Float32x arguments to
   narrowing functions.
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

#include <bits/floatn.h>
#include <float.h>

#define ARG_FUNC(function) function ## f32x
#define ARG_FLOAT _Float32x
#define ARG_PREFIX FLT32X
#define ARG_LIT(x) __f32x (x)
#define ARG_TYPE_STR "double"
#define FUNC_NARROW_SUFFIX f32x
