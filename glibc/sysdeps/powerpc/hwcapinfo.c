/* powerpc HWCAP/HWCAP2 and AT_PLATFORM data pre-processing.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#include <unistd.h>
#include <shlib-compat.h>
#include <dl-procinfo.h>

uint64_t __tcb_hwcap __attribute__ ((visibility ("hidden")));
uint32_t __tcb_platform __attribute__ ((visibility ("hidden")));

/* This function parses the HWCAP/HWCAP2 fields, adding the previous supported
   ISA bits, as well as converting the AT_PLATFORM string to a number.  This
   data is stored in two global variables that can be used later by the
   powerpc-specific code to store it into the TCB.  */
void
__tcb_parse_hwcap_and_convert_at_platform (void)
{

  uint64_t h1, h2;

  /* Read AT_PLATFORM string from auxv and convert it to a number.  */
  __tcb_platform = _dl_string_platform (GLRO (dl_platform));

  /* Read HWCAP and HWCAP2 from auxv.  */
  h1 = GLRO (dl_hwcap);
  h2 = GLRO (dl_hwcap2);

  /* hwcap contains only the latest supported ISA, the code checks which is
     and fills the previous supported ones.  */

  if (h2 & PPC_FEATURE2_ARCH_2_07)
    h1 |= PPC_FEATURE_ARCH_2_06
       | PPC_FEATURE_ARCH_2_05
       | PPC_FEATURE_POWER5_PLUS
       | PPC_FEATURE_POWER5
       | PPC_FEATURE_POWER4;
  else if (h1 & PPC_FEATURE_ARCH_2_06)
    h1 |= PPC_FEATURE_ARCH_2_05
       | PPC_FEATURE_POWER5_PLUS
       | PPC_FEATURE_POWER5
       | PPC_FEATURE_POWER4;
  else if (h1 & PPC_FEATURE_ARCH_2_05)
    h1 |= PPC_FEATURE_POWER5_PLUS
       | PPC_FEATURE_POWER5
       | PPC_FEATURE_POWER4;
  else if (h1 & PPC_FEATURE_POWER5_PLUS)
    h1 |= PPC_FEATURE_POWER5
       | PPC_FEATURE_POWER4;
  else if (h1 & PPC_FEATURE_POWER5)
    h1 |= PPC_FEATURE_POWER4;

  /* Consolidate both HWCAP and HWCAP2 into a single doubleword so that
     we can read both in a single load later.  */
  __tcb_hwcap = h2;
  __tcb_hwcap = (h1 << 32) | __tcb_hwcap;

}
#if IS_IN (rtld)
versioned_symbol (ld, __tcb_parse_hwcap_and_convert_at_platform, \
		  __parse_hwcap_and_convert_at_platform, GLIBC_2_23);
#endif

/* Export __parse_hwcap_and_convert_at_platform in libc.a.  This is used by
   GCC to make sure that the HWCAP/Platform bits are stored in the TCB when
   using __builtin_cpu_is()/__builtin_cpu_supports() in the static case.  */
#ifndef SHARED
weak_alias (__tcb_parse_hwcap_and_convert_at_platform, \
	    __parse_hwcap_and_convert_at_platform);
#endif
