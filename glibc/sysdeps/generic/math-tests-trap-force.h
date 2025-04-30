/* Configuration for math tests: support for setting exception flags
   without causing enabled traps.  Generic version.
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

#ifndef _MATH_TESTS_TRAP_FORCE_H
#define _MATH_TESTS_TRAP_FORCE_H 1

/* Indicate whether exception traps, if enabled, occur whenever an
   exception flag is set explicitly, so it is not possible to set flag
   bits with traps enabled without causing traps to be taken.  If
   traps cannot be enabled, the value of this macro does not
   matter.  */
#define EXCEPTION_SET_FORCES_TRAP 0

#endif /* math-tests-trap-force.h.  */
