/* Configuration for math tests: sNaN payloads.  Generic version.
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

#ifndef _MATH_TESTS_SNAN_PAYLOAD_H
#define _MATH_TESTS_SNAN_PAYLOAD_H 1

/* Indicate whether operations on signaling NaNs preserve the payload
   (if possible; it is not possible with a zero payload if the high
   bit is set for signaling NaNs) when generating a quiet NaN, and
   this should be tested.  */
#define SNAN_TESTS_PRESERVE_PAYLOAD	1

#endif /* math-tests-snan-payload.h.  */
