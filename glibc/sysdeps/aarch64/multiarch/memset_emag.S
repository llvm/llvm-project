/* Optimized memset for AmpereComputing emag processor.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.

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

#include <sysdep.h>

#if IS_IN (libc)
# define MEMSET __memset_emag

/*
 * Using DC ZVA to zero memory does not produce better performance if
 * memory size is not very large, especially when there are multiple
 * processes/threads contending memory/cache. Here we set threshold to
 * zero to disable using DC ZVA, which is good for multi-process/thread
 * workloads.
 */

# define DC_ZVA_THRESHOLD 0

# include "./memset_base64.S"
#endif
