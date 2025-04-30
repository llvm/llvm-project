/* gmp-mparam.h -- Compiler/machine parameter header file.
Copyright (C) 1991-2021 Free Software Foundation, Inc.

This file is part of the GNU MP Library.

The GNU MP Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 2.1 of the License, or (at your
option) any later version.

The GNU MP Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the GNU MP Library; see the file COPYING.LIB.  If not, see
<https://www.gnu.org/licenses/>.  */

#include <bits/wordsize.h>

#define BITS_PER_MP_LIMB	__WORDSIZE
#define BYTES_PER_MP_LIMB	(__WORDSIZE / 8)
#define BITS_PER_LONGINT	__WORDSIZE
#define BITS_PER_INT		32
#define BITS_PER_SHORTINT	16
#define BITS_PER_CHAR		8
