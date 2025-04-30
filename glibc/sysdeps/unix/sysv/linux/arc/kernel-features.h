/* Set flags signalling availability of kernel features based on given
   kernel version number.

   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

/* The minimum supported kernel version for ARC is 5.1 (64-bit time, offsets),
   although the asm-generic ABI is from 3.9 (when Linux port was merged).  */

#include_next <kernel-features.h>

#undef __ASSUME_CLONE_DEFAULT
#define __ASSUME_CLONE_BACKWARDS 1
