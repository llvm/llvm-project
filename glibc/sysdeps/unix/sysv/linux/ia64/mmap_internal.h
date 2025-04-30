/* Common mmap definition for Linux implementation.  Linux/ia64 version.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#ifndef MMAP_IA64_INTERNAL_LINUX_H
#define MMAP_IA64_INTERNAL_LINUX_H

/* Linux allows PAGE_SHIFT in range of [12-16] and expect
   mmap2 offset to be provided in based on the configured pagesize.
   Determine the shift dynamically with getpagesize.  */
#define MMAP2_PAGE_UNIT -1

#include_next <mmap_internal.h>

#endif
