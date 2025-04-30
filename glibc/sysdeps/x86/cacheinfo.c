/* x86 cache info.
   Copyright (C) 2003-2021 Free Software Foundation, Inc.
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

#if IS_IN (libc)

#include <unistd.h>
#include <ldsodefs.h>

/* Get the value of the system variable NAME.  */
long int
attribute_hidden
__cache_sysconf (int name)
{
  const struct cpu_features *cpu_features = __get_cpu_features ();
  switch (name)
    {
    case _SC_LEVEL1_ICACHE_SIZE:
      return cpu_features->level1_icache_size;

    case _SC_LEVEL1_ICACHE_LINESIZE:
      return cpu_features->level1_icache_linesize;

    case _SC_LEVEL1_DCACHE_SIZE:
      return cpu_features->level1_dcache_size;

    case _SC_LEVEL1_DCACHE_ASSOC:
      return cpu_features->level1_dcache_assoc;

    case _SC_LEVEL1_DCACHE_LINESIZE:
      return cpu_features->level1_dcache_linesize;

    case _SC_LEVEL2_CACHE_SIZE:
      return cpu_features->level2_cache_size;

    case _SC_LEVEL2_CACHE_ASSOC:
      return cpu_features->level2_cache_assoc;

    case _SC_LEVEL2_CACHE_LINESIZE:
      return cpu_features->level2_cache_linesize;

    case _SC_LEVEL3_CACHE_SIZE:
      return cpu_features->level3_cache_size;

    case _SC_LEVEL3_CACHE_ASSOC:
      return cpu_features->level3_cache_assoc;

    case _SC_LEVEL3_CACHE_LINESIZE:
      return cpu_features->level3_cache_linesize;

    case _SC_LEVEL4_CACHE_SIZE:
      return cpu_features->level4_cache_size;

    default:
      break;
    }
  return -1;
}

# ifdef SHARED
/* NB: In libc.a, cacheinfo.h is included in libc-start.c.  In libc.so,
   cacheinfo.h is included here and call init_cacheinfo by initializing
   a dummy function pointer via IFUNC relocation after CPU features in
   ld.so have been initialized by DL_PLATFORM_INIT or IFUNC relocation.  */
# include <cacheinfo.h>
# include <ifunc-init.h>

extern void __x86_cacheinfo (void) attribute_hidden;
void (*const __x86_cacheinfo_p) (void) attribute_hidden
  = __x86_cacheinfo;

__ifunc (__x86_cacheinfo, __x86_cacheinfo, NULL, void, init_cacheinfo);
# endif
#endif
