/* x86 version of hardware capability information handling macros.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.

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

#ifndef _DL_HWCAP_H
#define _DL_HWCAP_H

#if IS_IN (ldconfig)
/* Since ldconfig processes both i386 and x86-64 libraries, it needs
   to cover all platforms and hardware capabilities.  */
# define HWCAP_PLATFORMS_START	0
# define HWCAP_PLATFORMS_COUNT	4
# define HWCAP_START		0
# define HWCAP_COUNT		3
# define HWCAP_IMPORTANT \
  (HWCAP_X86_SSE2 | HWCAP_X86_64 | HWCAP_X86_AVX512_1)
#elif defined __x86_64__
/* For 64 bit, only cover x86-64 platforms and capabilities.  */
# define HWCAP_PLATFORMS_START	2
# define HWCAP_PLATFORMS_COUNT	4
# define HWCAP_START		1
# define HWCAP_COUNT		3
# define HWCAP_IMPORTANT	(HWCAP_X86_64 | HWCAP_X86_AVX512_1)
#else
/* For 32 bit, only cover i586, i686 and SSE2.  */
# define HWCAP_PLATFORMS_START	0
# define HWCAP_PLATFORMS_COUNT	2
# define HWCAP_START		0
# define HWCAP_COUNT		1
# define HWCAP_IMPORTANT	(HWCAP_X86_SSE2)
#endif

enum
{
  HWCAP_X86_SSE2		= 1 << 0,
  HWCAP_X86_64			= 1 << 1,
  HWCAP_X86_AVX512_1		= 1 << 2
};

static inline const char *
__attribute__ ((unused))
_dl_hwcap_string (int idx)
{
  return GLRO(dl_x86_hwcap_flags)[idx];
};

static inline int
__attribute__ ((unused, always_inline))
_dl_string_hwcap (const char *str)
{
  int i;

  for (i = HWCAP_START; i < HWCAP_COUNT; i++)
    {
      if (strcmp (str, GLRO(dl_x86_hwcap_flags)[i]) == 0)
	return i;
    }
  return -1;
};

/* We cannot provide a general printing function.  */
#define _dl_procinfo(type, word) -1

#endif /* dl-hwcap.h */
