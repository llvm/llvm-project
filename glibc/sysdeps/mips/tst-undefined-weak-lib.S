/* Undefined weak symbol loading shared module.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

#include <sgidefs.h>

    .text
    .globl	x
    .set	nomips16
    .set	nomicromips
    .ent	x
    .type	x, @function
x:
    .set noreorder
#if _MIPS_SIM == _ABIO32
    .cpload  $25
    jr  $31
    lb  $2,%got(a)($28)
#else
    .cpsetup  $25,$24,x
    lb  $2,%got_disp(a)($28)
    jr  $31
    .cpreturn
#endif
    .set reorder
    .end	x
    .size	x, .-x
    .weak	a
    .hidden	a
