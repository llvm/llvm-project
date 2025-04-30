/* Copyright (C) 2020-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <errno.h>
#include <unistd.h>
#include <sys/auxv.h>
#include <stdbool.h>

static long int linux_sysconf (int name);

static inline bool
getauxval2_einval (unsigned long int type, unsigned long int *result)
{
  if (__getauxval2 (type, result))
    return true;

  __set_errno (EINVAL);

  return false;
}

static inline long int
sysconf_get_cache_associativity (unsigned long type)
{
  unsigned long int result;

  if (getauxval2_einval (type, &result))
    return (result & 0xffff0000) >> 16;

  return -1;
}

static inline long int
sysconf_get_cache_linesize (unsigned long type)
{
  unsigned long int result;

  if (getauxval2_einval (type, &result))
    return result & 0xffff;

  return -1;
}

static inline long int
sysconf_get_cache_size (unsigned long type)
{
  unsigned long int result;

  if (getauxval2_einval (type, &result))
    return result;

  return -1;
}

/* Get the value of the system variable NAME.  */
long int
__sysconf (int name)
{
  switch (name)
    {
      case _SC_LEVEL1_ICACHE_SIZE:
	return sysconf_get_cache_size (AT_L1I_CACHESIZE);
      case _SC_LEVEL1_ICACHE_ASSOC:
	return sysconf_get_cache_associativity (AT_L1I_CACHEGEOMETRY);
      case _SC_LEVEL1_ICACHE_LINESIZE:
	return sysconf_get_cache_linesize (AT_L1I_CACHEGEOMETRY);
      case _SC_LEVEL1_DCACHE_SIZE:
	return sysconf_get_cache_size (AT_L1D_CACHESIZE);
      case _SC_LEVEL1_DCACHE_ASSOC:
	return sysconf_get_cache_associativity (AT_L1D_CACHEGEOMETRY);
      case _SC_LEVEL1_DCACHE_LINESIZE:
	return sysconf_get_cache_linesize (AT_L1D_CACHEGEOMETRY);
      case _SC_LEVEL2_CACHE_SIZE:
	return sysconf_get_cache_size (AT_L2_CACHESIZE);
      case _SC_LEVEL2_CACHE_ASSOC:
	return sysconf_get_cache_associativity (AT_L2_CACHEGEOMETRY);
      case _SC_LEVEL2_CACHE_LINESIZE:
	return sysconf_get_cache_linesize (AT_L2_CACHEGEOMETRY);
      default:
	return linux_sysconf (name);
    }
}

/* Now the generic Linux version.  */
#undef __sysconf
#define __sysconf static linux_sysconf
#include <sysdeps/unix/sysv/linux/sysconf.c>
