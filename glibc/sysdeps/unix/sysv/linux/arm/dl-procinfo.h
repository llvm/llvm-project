/* Linux/ARM version of processor capability information handling macros.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Philip Blundell <philb@gnu.org>, 2001.

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
#include <sysdep.h>

#define _DL_HWCAP_COUNT 27

/* Low 22 bits are allocated in HWCAP.  */
#define _DL_HWCAP_LAST		21

/* Low 5 bits are allocated in HWCAP2.  */
#define _DL_HWCAP2_LAST		4

/* The kernel provides platform data but it is not interesting.  */
#define _DL_HWCAP_PLATFORM	0


static inline const char *
__attribute__ ((unused))
_dl_hwcap_string (int idx)
{
  return GLRO(dl_arm_cap_flags)[idx];
};

static inline int
__attribute__ ((unused))
_dl_procinfo (unsigned int type, unsigned long int word)
{
  switch(type)
    {
    case AT_HWCAP:
      _dl_printf ("AT_HWCAP:       ");

      for (int i = 0; i <= _DL_HWCAP_LAST; ++i)
	if (word & (1 << i))
	  _dl_printf (" %s", _dl_hwcap_string (i));
      break;
    case AT_HWCAP2:
      {
	unsigned int offset = _DL_HWCAP_LAST + 1;

	_dl_printf ("AT_HWCAP2:      ");

	for (int i = 0; i <= _DL_HWCAP2_LAST; ++i)
	  if (word & (1 << i))
	    _dl_printf (" %s", _dl_hwcap_string (offset + i));
	break;
      }
    default:
      /* Fallback to generic output mechanism.  */
      return -1;
    }
  _dl_printf ("\n");
  return 0;
}

#define HWCAP_IMPORTANT		(HWCAP_ARM_VFP | HWCAP_ARM_NEON)

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

#define _dl_string_platform(str) (-1)

#endif /* dl-procinfo.h */
