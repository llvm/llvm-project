/* s390 version of processor capability information handling macros.
   Copyright (C) 2006-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Martin Schwidefsky <schwidefsky@de.ibm.com>, 2006.

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

#ifndef _DL_PROCINFO_H
#define _DL_PROCINFO_H	1
#include <ldsodefs.h>

#define _DL_HWCAP_COUNT 21

#define _DL_PLATFORMS_COUNT	10

/* The kernel provides up to 32 capability bits with elf_hwcap.  */
#define _DL_FIRST_PLATFORM	32
/* Mask to filter out platforms.  */
#define _DL_HWCAP_PLATFORM	(((1ULL << _DL_PLATFORMS_COUNT) - 1) \
				 << _DL_FIRST_PLATFORM)

/* Hardware capability bit numbers are derived directly from the
   facility indications as stored by the "store facility list" (STFL)
   instruction.
   highgprs is an alien in that list.  It describes a *kernel*
   capability.  */

enum
{
  HWCAP_S390_ESAN3 = 1 << 0,
  HWCAP_S390_ZARCH = 1 << 1,
  HWCAP_S390_STFLE = 1 << 2,
  HWCAP_S390_MSA = 1 << 3,
  HWCAP_S390_LDISP = 1 << 4,
  HWCAP_S390_EIMM = 1 << 5,
  HWCAP_S390_DFP = 1 << 6,
  HWCAP_S390_HPAGE = 1 << 7,
  HWCAP_S390_ETF3EH = 1 << 8,
  HWCAP_S390_HIGH_GPRS = 1 << 9,
  HWCAP_S390_TE = 1 << 10,
  HWCAP_S390_VX = 1 << 11,
  HWCAP_S390_VXRS = HWCAP_S390_VX,
  HWCAP_S390_VXD = 1 << 12,
  HWCAP_S390_VXRS_BCD = HWCAP_S390_VXD,
  HWCAP_S390_VXE = 1 << 13,
  HWCAP_S390_VXRS_EXT = HWCAP_S390_VXE,
  HWCAP_S390_GS = 1 << 14,
  HWCAP_S390_VXRS_EXT2 = 1 << 15,
  HWCAP_S390_VXRS_PDE = 1 << 16,
  HWCAP_S390_SORT = 1 << 17,
  HWCAP_S390_DFLT = 1 << 18,
  HWCAP_S390_VXRS_PDE2 = 1 << 19,
  HWCAP_S390_NNPA = 1 << 20,
};

#define HWCAP_IMPORTANT (HWCAP_S390_ZARCH | HWCAP_S390_LDISP \
			 | HWCAP_S390_EIMM | HWCAP_S390_DFP  \
			 | HWCAP_S390_VX | HWCAP_S390_VXE    \
			 | HWCAP_S390_VXRS_EXT2)

/* We cannot provide a general printing function.  */
#define _dl_procinfo(type, word) -1

static inline const char *
__attribute__ ((unused))
_dl_hwcap_string (int idx)
{
  return GLRO(dl_s390_cap_flags)[idx];
};

static inline int
__attribute__ ((unused, always_inline))
_dl_string_hwcap (const char *str)
{
  int i;

  for (i = 0; i < _DL_HWCAP_COUNT; i++)
    {
      if (strcmp (str, GLRO(dl_s390_cap_flags)[i]) == 0)
	return i;
    }
  return -1;
};

static inline int
__attribute__ ((unused, always_inline))
_dl_string_platform (const char *str)
{
  int i;

  if (str != NULL)
    for (i = 0; i < _DL_PLATFORMS_COUNT; ++i)
      {
	if (strcmp (str, GLRO(dl_s390_platforms)[i]) == 0)
	  return _DL_FIRST_PLATFORM + i;
      }
  return -1;
};

#endif /* dl-procinfo.h */
