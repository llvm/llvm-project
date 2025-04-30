/* Configuration for math tests: support for enabling exception traps.
   Generic version.
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
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

#ifndef _MATH_TESTS_TRAP_H
#define _MATH_TESTS_TRAP_H 1

/* Indicate whether the given exception trap(s) can be enabled in
   feenableexcept.  If non-zero, the traps are always supported.  If
   zero, traps may or may not be supported depending on the target
   (this can be determined by checking the return value of
   feenableexcept).  This enables skipping of tests which use traps.
   By default traps are supported unless overridden.  */
#define EXCEPTION_ENABLE_SUPPORTED(EXCEPT)		\
  (EXCEPTION_TESTS_float || EXCEPTION_TESTS_double)

#endif /* math-tests-trap.h.  */
