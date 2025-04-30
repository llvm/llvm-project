/* Architecture-specific glibc-hwcaps subdirectories.  x86 version.
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
#include <cpu-features.h>
#include <ldsodefs.h>
#include <get-isa-level.h>

const char _dl_hwcaps_subdirs[] = "x86-64-v4:x86-64-v3:x86-64-v2";
enum { subdirs_count = 3 }; /* Number of components in _dl_hwcaps_subdirs.  */

uint32_t
_dl_hwcaps_subdirs_active (void)
{
  const struct cpu_features *cpu_features = __get_cpu_features ();
  unsigned int isa_level = get_isa_level (cpu_features);
  int active = 0;

  /* Test in reverse preference order.  */

  /* x86-64-v2.  */
  if (!(isa_level & GNU_PROPERTY_X86_ISA_1_V2))
    return _dl_hwcaps_subdirs_build_bitmask (subdirs_count, active);
  ++active;

  /* x86-64-v3.  */
  if (!(isa_level & GNU_PROPERTY_X86_ISA_1_V3))
    return _dl_hwcaps_subdirs_build_bitmask (subdirs_count, active);
  ++active;

  /* x86-64-v4.  */
  if (!(isa_level & GNU_PROPERTY_X86_ISA_1_V4))
    return _dl_hwcaps_subdirs_build_bitmask (subdirs_count, active);
  ++active;

  return _dl_hwcaps_subdirs_build_bitmask (subdirs_count, active);
}
