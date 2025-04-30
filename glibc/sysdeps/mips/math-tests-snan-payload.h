/* Configuration for math tests: sNaN payloads.  MIPS version.
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

#ifndef MIPS_MATH_TESTS_SNAN_PAYLOAD_H
#define MIPS_MATH_TESTS_SNAN_PAYLOAD_H 1

/* NaN payload preservation when converting a signaling NaN to quiet
   is only required in NAN2008 mode.  */
#ifdef __mips_nan2008
# define SNAN_TESTS_PRESERVE_PAYLOAD	1
#else
# define SNAN_TESTS_PRESERVE_PAYLOAD	0
#endif

#endif /* math-tests-snan-payload.h.  */
