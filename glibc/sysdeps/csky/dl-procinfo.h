/* C-SKY version of processor capability information handling macros.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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


#ifndef _DL_PROCINFO_H
#define _DL_PROCINFO_H	1

#include <ldsodefs.h>

/* Mask to filter out platforms.  */
#define _DL_HWCAP_PLATFORM    (-1ULL)

#define _DL_PLATFORMS_COUNT   4

static inline int
__attribute__ ((unused, always_inline))
_dl_string_platform (const char *str)
{
  int i;

  if (str != NULL)
    for (i = 0; i < _DL_PLATFORMS_COUNT; ++i)
      {
        if (strcmp (str, GLRO(dl_csky_platforms)[i]) == 0)
          return i;
      }
  return -1;
};

/* We cannot provide a general printing function.  */
#define _dl_procinfo(word, val) -1

/* There are no hardware capabilities defined.  */
#define _dl_hwcap_string(idx) ""

/* By default there is no important hardware capability.  */
#define HWCAP_IMPORTANT (0)

/* We don't have any hardware capabilities.  */
#define _DL_HWCAP_COUNT	0

#define _dl_string_hwcap(str) (-1)

#endif /* dl-procinfo.h */
