/* Check for hardware capabilities after HWCAP parsing.  S390 version.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

#ifndef _DL_HWCAP_CHECK_H
#define _DL_HWCAP_CHECK_H

#include <ldsodefs.h>

static inline void
dl_hwcap_check (void)
{
#if defined __ARCH__
# if __ARCH__ >= 13
  if (!(GLRO(dl_hwcap) & HWCAP_S390_VXRS_EXT2))
    _dl_fatal_printf ("\
Fatal glibc error: CPU lacks VXRS_EXT2 support (z15 or later required)\n");
# elif __ARCH__ >= 12
  if (!(GLRO(dl_hwcap) & HWCAP_S390_VXE))
    _dl_fatal_printf ("\
Fatal glibc error: CPU lacks VXE support (z14 or later required)\n");
# endif
#endif /* __ARCH__ */
}

#endif /* _DL_HWCAP_CHECK_H */
