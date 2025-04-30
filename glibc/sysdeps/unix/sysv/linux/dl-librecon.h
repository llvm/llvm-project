/* Optional code to distinguish library flavours.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2001.

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

#ifndef _DL_LIBRECON_H
#define _DL_LIBRECON_H	1

static inline void __attribute__ ((unused, always_inline))
_dl_osversion_init (char *assume_kernel)
{
  unsigned long int i, j, osversion = 0;
  char *p = assume_kernel, *q;

  for (i = 0; i < 3; i++, p = q + 1)
    {
      j = _dl_strtoul (p, &q);
      if (j >= 255 || p == q || (i < 2 && *q && *q != '.'))
	{
	  osversion = 0;
	  break;
	}
      osversion |= j << (16 - 8 * i);
      if (!*q)
	break;
    }
  if (osversion)
    GLRO(dl_osversion) = osversion;
}

/* Recognizing extra environment variables.  */
#define EXTRA_LD_ENVVARS_13 \
    if (memcmp (envline, "ASSUME_KERNEL", 13) == 0)			      \
      {									      \
	_dl_osversion_init (&envline[14]);				      \
	break;								      \
      }

#define DL_OSVERSION_INIT \
  do {									      \
    char *assume_kernel = getenv ("LD_ASSUME_KERNEL");			      \
    if (assume_kernel)							      \
      _dl_osversion_init (assume_kernel);				      \
  } while (0)

#endif /* dl-librecon.h */
