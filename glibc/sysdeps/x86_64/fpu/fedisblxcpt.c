/* Disable floating-point exceptions.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Andreas Jaeger <aj@suse.de>, 2001.

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

#include <fenv.h>

int
fedisableexcept (int excepts)
{
  unsigned short int new_exc, old_exc;
  unsigned int new;

  excepts &= FE_ALL_EXCEPT;

  /* Get the current control word of the x87 FPU.  */
  __asm__ ("fstcw %0" : "=m" (*&new_exc));

  old_exc = (~new_exc) & FE_ALL_EXCEPT;

  new_exc |= excepts;
  __asm__ ("fldcw %0" : : "m" (*&new_exc));

  /* And now the same for the SSE MXCSR register.  */
  __asm__ ("stmxcsr %0" : "=m" (*&new));

  /* The SSE exception masks are shifted by 7 bits.  */
  new |= excepts << 7;
  __asm__ ("ldmxcsr %0" : : "m" (*&new));

  return old_exc;
}
