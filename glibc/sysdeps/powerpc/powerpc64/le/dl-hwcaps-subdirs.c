/* Architecture-specific glibc-hwcaps subdirectories.  powerpc64le version.
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

#include <dl-hwcaps.h>
#include <ldsodefs.h>

const char _dl_hwcaps_subdirs[] = "power10:power9";
enum { subdirs_count = 2 }; /* Number of components in _dl_hwcaps_subdirs.  */

uint32_t
_dl_hwcaps_subdirs_active (void)
{
  int active = 0;

  /* Test in reverse preference order.  Altivec and VSX are implied by
     the powerpc64le ABI definition.  */

  /* POWER9.  GCC enables float128 hardware support for -mcpu=power9.  */
  if ((GLRO (dl_hwcap2) & PPC_FEATURE2_ARCH_3_00) == 0
      || (GLRO (dl_hwcap2) & PPC_FEATURE2_HAS_IEEE128) == 0)
    return _dl_hwcaps_subdirs_build_bitmask (subdirs_count, active);
  ++active;

  /* POWER10.  GCC defines __MMA__ for -mcpu=power10.  */
  if ((GLRO (dl_hwcap2) & PPC_FEATURE2_ARCH_3_1) == 0
      || (GLRO (dl_hwcap2) & PPC_FEATURE2_MMA) == 0)
    return _dl_hwcaps_subdirs_build_bitmask (subdirs_count, active);
  ++active;

  return _dl_hwcaps_subdirs_build_bitmask (subdirs_count, active);
}
