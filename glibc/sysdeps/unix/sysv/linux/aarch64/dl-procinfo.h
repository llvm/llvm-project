/* Processor capability information handling macros - aarch64 version.
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

#include <sys/auxv.h>
#include <unistd.h>
#include <ldsodefs.h>
#include <sysdep.h>

/* We cannot provide a general printing function.  */
#define _dl_procinfo(type, word) -1

/* No additional library search paths.  */
#define HWCAP_IMPORTANT HWCAP_ATOMICS

static inline const char *
__attribute__ ((unused))
_dl_hwcap_string (int idx)
{
  return (unsigned)idx < _DL_HWCAP_COUNT ? GLRO(dl_aarch64_cap_flags)[idx] : "";
};

static inline int
__attribute__ ((unused))
_dl_string_hwcap (const char *str)
{
  for (int i = 0; i < _DL_HWCAP_COUNT; i++)
    {
      if (strcmp (str, _dl_hwcap_string (i)) == 0)
	return i;
    }
  return -1;
};

/* There're no platforms to filter out.  */
#define _DL_HWCAP_PLATFORM 0

#define _dl_string_platform(str) (-1)

#endif /* dl-procinfo.h */
