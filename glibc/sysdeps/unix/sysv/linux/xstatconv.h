/* Convert between the kernel's `struct stat' format, and libc's.
   Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#if !STAT_IS_KERNEL_STAT
extern int __xstat_conv (int vers, struct kernel_stat *kbuf, void *ubuf)
  attribute_hidden;
extern int __xstat64_conv (int vers, struct kernel_stat *kbuf, void *ubuf)
  attribute_hidden;
#endif
extern int __xstat32_conv (int vers, struct stat64 *kbuf, struct stat *buf)
  attribute_hidden;
