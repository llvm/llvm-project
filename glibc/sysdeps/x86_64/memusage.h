/* Copyright (C) 2001-2021 Free Software Foundation, Inc.
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

#if defined(__clang__)
  #define GETSP() ({ uintptr_t f; asm("mov %%rsp, %0" : "=r"(f)); f; })
#else
  #define GETSP() ({ register uintptr_t stack_ptr asm ("rsp"); stack_ptr; })
#endif
#define GETTIME(low,high) asm ("rdtsc" : "=a" (low), "=d" (high))

#include <sysdeps/generic/memusage.h>
