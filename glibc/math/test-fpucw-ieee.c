/* FPU control word overridden initialization test.
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

#ifdef _FPU_IEEE
/* Some architectures don't have _FPU_IEEE.  */
# define FPU_CONTROL _FPU_IEEE
#endif

#include <test-fpucw.c>

/* Preempt the library's definition of `__fpu_control'.  */
fpu_control_t __fpu_control = FPU_CONTROL;
