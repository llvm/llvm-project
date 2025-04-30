/* Specify architecture-specific rules for determining tininess of
   floating-point results.  Generic version.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

#ifndef _TININESS_H
#define _TININESS_H	1

/* Under IEEE 754, an architecture may determine tininess of
   floating-point results either "before rounding" or "after
   rounding", but must do so in the same way for all operations
   returning binary results.  Define TININESS_AFTER_ROUNDING to 1 for
   "after rounding" architectures, 0 for "before rounding"
   architectures.  The test stdlib/tst-tininess will fail if the
   definition is incorrect.  */

#define TININESS_AFTER_ROUNDING	0

#endif /* tininess.h */
