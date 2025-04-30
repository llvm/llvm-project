/* Changing the per-thread memory protection key, powerpc64 version.
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
   <http://www.gnu.org/licenses/>.  */

#include <arch-pkey.h>
#include <errno.h>
#include <sys/mman.h>

int
pkey_set (int key, unsigned int rights)
{
  if (key < 0 || key > PKEY_MAX || rights > 3)
    {
      __set_errno (EINVAL);
      return -1;
    }

  /* Translate to AMR bit values.  */
  unsigned long int bits;
  if (rights & PKEY_DISABLE_ACCESS)
    /* The PKEY_DISABLE_WRITE bit does not matter.  */
    bits = PKEY_AMR_READ | PKEY_AMR_WRITE;
  else if (rights == PKEY_DISABLE_WRITE)
    bits = PKEY_AMR_WRITE;
  else
    bits = 0;

  unsigned int index = pkey_index (key);
  unsigned long int mask = 3UL << index;
  unsigned long int amr = pkey_read ();
  amr = (amr & ~mask) | (bits << index);
  pkey_write (amr);
  return 0;
}
