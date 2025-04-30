/* Return info on filesystem.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
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

#include <shlib-compat.h>

/* This deprecated syscall is no longer used (replaced with {f}statfs).  */
#if SHLIB_COMPAT(libc, GLIBC_2_0, GLIBC_2_28)

# include <unistd.h>
# include <errno.h>
# include <sys/types.h>

struct ustat
 {
   __daddr_t f_tfree;         /* Number of free blocks.  */
   __ino_t f_tinode;          /* Number of free inodes.  */
   char f_fname[6];
   char f_fpack[6];
};

int
__old_ustat (dev_t dev, struct ustat *ust)
{
  __set_errno (ENOSYS);
  return -1;
}
stub_warning (ustat)
compat_symbol (libc, __old_ustat, ustat, GLIBC_2_0);
#endif
