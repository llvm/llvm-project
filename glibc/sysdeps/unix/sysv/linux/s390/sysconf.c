/* Get system parameters, e.g. cache information.  S390/S390x version.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <unistd.h>
#include <dl-procinfo.h>

static long int linux_sysconf (int name);

/* Possible arguments for get_cache_info.
   The values are reflecting the level/attribute/type indications
   of ecag-instruction (extract cpu attribue).  */
#define CACHE_LEVEL_MAX        8
#define CACHE_ATTR_LINESIZE    1
#define CACHE_ATTR_SIZE        2
#define CACHE_ATTR_ASSOC       3
#define CACHE_TYPE_DATA        0
#define CACHE_TYPE_INSTRUCTION 1

static long
get_cache_info (int level, int attr, int type)
{
  unsigned long int val;
  unsigned int cmd;
  unsigned long int arg;

  /* Check arguments.  */
  if (level < 1 || level > CACHE_LEVEL_MAX
      || attr < CACHE_ATTR_LINESIZE || attr > CACHE_ATTR_ASSOC
      || type < CACHE_TYPE_DATA || type > CACHE_TYPE_INSTRUCTION)
    return 0L;

  /* Check if ecag-instruction is available.
     ecag - extract CPU attribute (only in zarch; arch >= z10; in as 2.24)  */
  if (!(GLRO (dl_hwcap) & HWCAP_S390_STFLE)
#if !defined __s390x__
      || !(GLRO (dl_hwcap) & HWCAP_S390_ZARCH)
      || !(GLRO (dl_hwcap) & HWCAP_S390_HIGH_GPRS)
#endif /* !__s390x__ */
      )
    {
      /* stfle (or zarch, high-gprs on s390-32) is not available.
	 We are on an old machine. Return 256byte for LINESIZE for L1 d/i-cache,
	 otherwise 0.  */
      if (level == 1 && attr == CACHE_ATTR_LINESIZE)
	return 256L;
      else
	return 0L;
    }

  /* Store facility list and check for z10.
     (see ifunc-resolver for details)  */
  register unsigned long reg0 __asm__("0") = 0;
#ifdef __s390x__
  unsigned long stfle_bits;
# define STFLE_Z10_MASK (1UL << (63 - 34))
#else
  unsigned long long stfle_bits;
# define STFLE_Z10_MASK (1ULL << (63 - 34))
#endif /* !__s390x__ */
  __asm__ __volatile__(".machine push"        "\n\t"
		       ".machinemode \"zarch_nohighgprs\"\n\t"
		       ".machine \"z9-109\""  "\n\t"
		       "stfle %0"             "\n\t"
		       ".machine pop"         "\n"
		       : "=QS" (stfle_bits), "+d" (reg0)
		       : : "cc");

  if (!(stfle_bits & STFLE_Z10_MASK))
    {
      /* We are at least on a z9 machine.
	 Return 256byte for LINESIZE for L1 d/i-cache,
	 otherwise 0.  */
      if (level == 1 && attr == CACHE_ATTR_LINESIZE)
	return 256L;
      else
	return 0L;
    }

  /* Check cache topology, if cache is available at this level.  */
  arg = (CACHE_LEVEL_MAX - level) * 8;
  __asm__ __volatile__ (".machine push\n\t"
			".machine \"z10\"\n\t"
			".machinemode \"zarch_nohighgprs\"\n\t"
			"ecag %0,%%r0,0\n\t"   /* returns 64bit unsigned integer.  */
			"srlg %0,%0,0(%1)\n\t" /* right align 8bit cache info field.  */
			".machine pop"
			: "=&d" (val)
			: "a" (arg)
			);
  val &= 0xCUL; /* Extract cache scope information from cache topology summary.
		   (bits 4-5 of 8bit-field; 00 means cache does not exist).  */
  if (val == 0)
    return 0L;

  /* Get cache information for level, attribute and type.  */
  cmd = (attr << 4) | ((level - 1) << 1) | type;
  __asm__ __volatile__ (".machine push\n\t"
			".machine \"z10\"\n\t"
			".machinemode \"zarch_nohighgprs\"\n\t"
			"ecag %0,%%r0,0(%1)\n\t"
			".machine pop"
			: "=d" (val)
			: "a" (cmd)
			);
  return val;
}

long int
__sysconf (int name)
{
  if (name >= _SC_LEVEL1_ICACHE_SIZE && name <= _SC_LEVEL4_CACHE_LINESIZE)
    {
      int level;
      int attr;
      int type;

      switch (name)
	{
	case _SC_LEVEL1_ICACHE_SIZE:
	  level = 1;
	  attr = CACHE_ATTR_SIZE;
	  type = CACHE_TYPE_INSTRUCTION;
	  break;
	case _SC_LEVEL1_ICACHE_ASSOC:
	  level = 1;
	  attr = CACHE_ATTR_ASSOC;
	  type = CACHE_TYPE_INSTRUCTION;
	  break;
	case _SC_LEVEL1_ICACHE_LINESIZE:
	  level = 1;
	  attr = CACHE_ATTR_LINESIZE;
	  type = CACHE_TYPE_INSTRUCTION;
	  break;

	case _SC_LEVEL1_DCACHE_SIZE:
	  level = 1;
	  attr = CACHE_ATTR_SIZE;
	  type = CACHE_TYPE_DATA;
	  break;
	case _SC_LEVEL1_DCACHE_ASSOC:
	  level = 1;
	  attr = CACHE_ATTR_ASSOC;
	  type = CACHE_TYPE_DATA;
	  break;
	case _SC_LEVEL1_DCACHE_LINESIZE:
	  level = 1;
	  attr = CACHE_ATTR_LINESIZE;
	  type = CACHE_TYPE_DATA;
	  break;

	case _SC_LEVEL2_CACHE_SIZE:
	  level = 2;
	  attr = CACHE_ATTR_SIZE;
	  type = CACHE_TYPE_DATA;
	  break;
	case _SC_LEVEL2_CACHE_ASSOC:
	  level = 2;
	  attr = CACHE_ATTR_ASSOC;
	  type = CACHE_TYPE_DATA;
	  break;
	case _SC_LEVEL2_CACHE_LINESIZE:
	  level = 2;
	  attr = CACHE_ATTR_LINESIZE;
	  type = CACHE_TYPE_DATA;
	  break;

	case _SC_LEVEL3_CACHE_SIZE:
	  level = 3;
	  attr = CACHE_ATTR_SIZE;
	  type = CACHE_TYPE_DATA;
	  break;
	case _SC_LEVEL3_CACHE_ASSOC:
	  level = 3;
	  attr = CACHE_ATTR_ASSOC;
	  type = CACHE_TYPE_DATA;
	  break;
	case _SC_LEVEL3_CACHE_LINESIZE:
	  level = 3;
	  attr = CACHE_ATTR_LINESIZE;
	  type = CACHE_TYPE_DATA;
	  break;

	case _SC_LEVEL4_CACHE_SIZE:
	  level = 4;
	  attr = CACHE_ATTR_SIZE;
	  type = CACHE_TYPE_DATA;
	  break;
	case _SC_LEVEL4_CACHE_ASSOC:
	  level = 4;
	  attr = CACHE_ATTR_ASSOC;
	  type = CACHE_TYPE_DATA;
	  break;
	case _SC_LEVEL4_CACHE_LINESIZE:
	  level = 4;
	  attr = CACHE_ATTR_LINESIZE;
	  type = CACHE_TYPE_DATA;
	  break;

	default:
	  level = 0;
	  attr = 0;
	  type = 0;
	  break;
	}

      return get_cache_info (level, attr, type);
    }

  return linux_sysconf (name);
}

/* Now the generic Linux version.  */
#undef __sysconf
#define __sysconf static linux_sysconf
#include <sysdeps/unix/sysv/linux/sysconf.c>
