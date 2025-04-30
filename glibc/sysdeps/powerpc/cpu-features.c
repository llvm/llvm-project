/* Initialize cpu feature data.  PowerPC version.
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

#include <stdint.h>
#include <cpu-features.h>

#if HAVE_TUNABLES
# include <elf/dl-tunables.h>
#endif

static inline void
init_cpu_features (struct cpu_features *cpu_features)
{
  /* Default is to use aligned memory access on optimized function unless
     tunables is enable, since for this case user can explicit disable
     unaligned optimizations.  */
#if HAVE_TUNABLES
  int32_t cached_memfunc = TUNABLE_GET (glibc, cpu, cached_memopt, int32_t,
					NULL);
  cpu_features->use_cached_memopt = (cached_memfunc > 0);
#else
  cpu_features->use_cached_memopt = false;
#endif
}
