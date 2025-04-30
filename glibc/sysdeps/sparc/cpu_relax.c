/* CPU strand yielding for busy loops.  Linux/sparc version.
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

#include <sparc-ifunc.h>

#ifdef __sparc_v9__
static void
__cpu_relax_generic (void)
{
  asm volatile ("rd %ccr, %g0;"
		"rd %ccr, %g0;"
		"rd %ccr, %g0");
}

static void
__cpu_relax_pause (void)
{
  asm volatile ("wr %g0, 128, %asr27");
}

sparc_libc_ifunc (__cpu_relax,
		  hwcap & HWCAP_SPARC_PAUSE
		  ? __cpu_relax_pause
		  : __cpu_relax_generic)
#endif
