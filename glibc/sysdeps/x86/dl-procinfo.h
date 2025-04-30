/* x86 version of processor capability information handling macros.
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

#ifndef _DL_PROCINFO_H
#define _DL_PROCINFO_H	1
#include <ldsodefs.h>
#include <dl-hwcap.h>

#define _DL_HWCAP_COUNT		HWCAP_COUNT
#define _DL_PLATFORMS_COUNT	HWCAP_PLATFORMS_COUNT

/* Start at 48 to reserve spaces for hardware capabilities.  */
#define _DL_FIRST_PLATFORM	48
/* Mask to filter out platforms.  */
#define _DL_HWCAP_PLATFORM	(((1ULL << _DL_PLATFORMS_COUNT) - 1) \
				 << _DL_FIRST_PLATFORM)

static inline int
__attribute__ ((unused, always_inline))
_dl_string_platform (const char *str)
{
  int i;

  if (str != NULL)
    for (i = HWCAP_PLATFORMS_START; i < HWCAP_PLATFORMS_COUNT; ++i)
      {
	if (strcmp (str, GLRO(dl_x86_platforms)[i]) == 0)
	  return _DL_FIRST_PLATFORM + i;
      }
  return -1;
};

#endif /* dl-procinfo.h */
