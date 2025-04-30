/* Helper functions for manipulating memory protection keys, for powerpc64.
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

#ifndef _ARCH_PKEY_H
#define _ARCH_PKEY_H

/* Read and write access bits in the AMR register.  Needs to be
   translated from and to PKEY_DISABLE_* flags.  */
#define PKEY_AMR_READ 1UL
#define PKEY_AMR_WRITE 2UL

/* Return the value of the AMR register.  */
static inline unsigned long int
pkey_read (void)
{
  unsigned long int result;
  __asm__ volatile ("mfspr %0, 13" : "=r" (result));
  return result;
}

/* Overwrite the AMR register with VALUE.  */
static inline void
pkey_write (unsigned long int value)
{
  __asm__ volatile ("isync; mtspr 13, %0; isync" : : "r" (value));
}

/* Number of the largest supported key.  This depends on the width of
   the AMR register.  */
#define PKEY_MAX (sizeof (unsigned long int) * 8 / 2 - 1)
_Static_assert (PKEY_MAX == 15 || PKEY_MAX == 31, "PKEY_MAX value");

/* Translate key number into AMR index position.  */
static inline int
pkey_index (int key)
{
  return 2 * (PKEY_MAX - key);
}

#endif /* _ARCH_PKEY_H */
