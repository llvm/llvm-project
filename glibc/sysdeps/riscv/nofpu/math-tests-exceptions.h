/* Configuration for math tests: support for exceptions.  RISC-V
   no-FPU version.
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

#ifndef RISCV_NOFPU_MATH_TESTS_EXCEPTIONS_H
#define RISCV_NOFPU_MATH_TESTS_EXCEPTIONS_H 1

/* We support setting floating-point exception flags on hard-float
   targets.  These are not supported on soft-float targets.  */
#define EXCEPTION_TESTS_float 0
#define EXCEPTION_TESTS_double        0
#define EXCEPTION_TESTS_long_double   0

#endif /* math-tests-exceptions.h.  */
