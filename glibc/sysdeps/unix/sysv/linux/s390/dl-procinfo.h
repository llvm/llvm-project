/* Linux/s390 version of processor capability information handling macros.
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

#ifndef _DL_S390_PROCINFO_H
#define _DL_S390_PROCINFO_H	1
#include <sysdeps/s390/dl-procinfo.h>
#include <ldsodefs.h>


#undef _dl_procinfo
static inline int
__attribute__ ((unused))
_dl_procinfo (unsigned int type, unsigned long int word)
{
  /* This table should match the information from arch/s390/kernel/setup.c
     in the kernel sources.  */
  int i;

  /* Fallback to generic output mechanism.  */
  if (type != AT_HWCAP)
    return -1;

  _dl_printf ("AT_HWCAP:   ");

  for (i = 0; i < _DL_HWCAP_COUNT; ++i)
    if (word & (1UL << i))
      _dl_printf (" %s", GLRO(dl_s390_cap_flags)[i]);

  _dl_printf ("\n");

  return 0;
}
#endif
