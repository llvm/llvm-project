/* Check for hardware capabilities after HWCAP parsing.  powerpc64le version.
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
#ifdef _ARCH_PWR9
  if ((GLRO (dl_hwcap2) & PPC_FEATURE2_ARCH_3_00) == 0)
    _dl_fatal_printf ("\
Fatal glibc error: CPU lacks ISA 3.00 support (POWER9 or later required)\n");
#endif
#ifdef __FLOAT128_HARDWARE__
  if ((GLRO (dl_hwcap2) & PPC_FEATURE2_HAS_IEEE128) == 0)
    _dl_fatal_printf ("\
Fatal glibc error: CPU lacks float128 support (POWER 9 or later required)\n");
#endif
   /* This check is not actually reached when building for POWER10 and
      running on POWER9 because there are faulting PCREL instructions
      before this point.  */
#if defined _ARCH_PWR10 || defined __PCREL__
  if ((GLRO (dl_hwcap2) & PPC_FEATURE2_ARCH_3_1) == 0)
    _dl_fatal_printf ("\
Fatal glibc error: CPU lacks ISA 3.10 support (POWER10 or later required)\n");
#endif
#ifdef __MMA__
  if ((GLRO (dl_hwcap2) & PPC_FEATURE2_MMA) == 0)
    _dl_fatal_printf ("\
Fatal glibc error: CPU lacks MMA support (POWER10 or later required)\n");
#endif
}

#endif /* _DL_HWCAP_CHECK_H */
