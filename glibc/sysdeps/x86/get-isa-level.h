/* Get x86 ISA level.
   This file is part of the GNU C Library.
   Copyright (C) 2020 Free Software Foundation, Inc.

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

#include <elf.h>
#include <cpu-features.h>

/* Get GNU_PROPERTY_X86_ISA_1_BASELINE and GNU_PROPERTY_X86_ISA_1_V[234]
   ISA level.  */

static unsigned int
get_isa_level (const struct cpu_features *cpu_features)
{
  unsigned int isa_level = 0;

  if (CPU_FEATURE_USABLE_P (cpu_features, CMOV)
      && CPU_FEATURE_USABLE_P (cpu_features, CX8)
      && CPU_FEATURE_CPU_P (cpu_features, FPU)
      && CPU_FEATURE_USABLE_P (cpu_features, FXSR)
      && CPU_FEATURE_USABLE_P (cpu_features, MMX)
      && CPU_FEATURE_USABLE_P (cpu_features, SSE)
      && CPU_FEATURE_USABLE_P (cpu_features, SSE2))
    {
      isa_level = GNU_PROPERTY_X86_ISA_1_BASELINE;
      if (CPU_FEATURE_USABLE_P (cpu_features, CMPXCHG16B)
	  && CPU_FEATURE_USABLE_P (cpu_features, LAHF64_SAHF64)
	  && CPU_FEATURE_USABLE_P (cpu_features, POPCNT)
	  && CPU_FEATURE_USABLE_P (cpu_features, SSE3)
	  && CPU_FEATURE_USABLE_P (cpu_features, SSSE3)
	  && CPU_FEATURE_USABLE_P (cpu_features, SSE4_1)
	  && CPU_FEATURE_USABLE_P (cpu_features, SSE4_2))
	{
	  isa_level |= GNU_PROPERTY_X86_ISA_1_V2;
	  if (CPU_FEATURE_USABLE_P (cpu_features, AVX)
	      && CPU_FEATURE_USABLE_P (cpu_features, AVX2)
	      && CPU_FEATURE_USABLE_P (cpu_features, F16C)
	      && CPU_FEATURE_USABLE_P (cpu_features, FMA)
	      && CPU_FEATURE_USABLE_P (cpu_features, LZCNT)
	      && CPU_FEATURE_USABLE_P (cpu_features, MOVBE))
	    {
	      isa_level |= GNU_PROPERTY_X86_ISA_1_V3;
	      if (CPU_FEATURE_USABLE_P (cpu_features, AVX512F)
		  && CPU_FEATURE_USABLE_P (cpu_features, AVX512BW)
		  && CPU_FEATURE_USABLE_P (cpu_features, AVX512CD)
		  && CPU_FEATURE_USABLE_P (cpu_features, AVX512DQ)
		  && CPU_FEATURE_USABLE_P (cpu_features, AVX512VL))
		isa_level |= GNU_PROPERTY_X86_ISA_1_V4;
	    }
	}
    }

  return isa_level;
}
