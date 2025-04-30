/* Common mmap definition for Linux implementation.  MIPS n32 version.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef MMAP_MIPS_N32_INTERNAL_H
#define MMAP_MIPS_N32_INTERNAL_H

/* To handle negative offsets consistently with other architectures,
   the offset must be zero-extended to 64-bit.  */
#define MMAP_ADJUST_OFFSET(offset) (uint64_t) (uint32_t) offset

#include_next <mmap_internal.h>

#endif
