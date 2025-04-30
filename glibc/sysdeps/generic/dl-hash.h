/* Compute hash value for given string according to ELF standard.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
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

#ifndef _DL_HASH_H
#define _DL_HASH_H	1


/* This is the hashing function specified by the ELF ABI.  In the
   first five operations no overflow is possible so we optimized it a
   bit.  */
static unsigned int
__attribute__ ((unused))
_dl_elf_hash (const char *name_arg)
{
  const unsigned char *name = (const unsigned char *) name_arg;
  unsigned long int hash = *name;
  if (hash != 0 && name[1] != '\0')
    {
      hash = (hash << 4) + name[1];
      if (name[2] != '\0')
	{
	  hash = (hash << 4) + name[2];
	  if (name[3] != '\0')
	    {
	      hash = (hash << 4) + name[3];
	      if (name[4] != '\0')
		{
		  hash = (hash << 4) + name[4];
		  name += 5;
		  while (*name != '\0')
		    {
		      unsigned long int hi;
		      hash = (hash << 4) + *name++;
		      hi = hash & 0xf0000000;

		      /* The algorithm specified in the ELF ABI is as
			 follows:

			 if (hi != 0)
			   hash ^= hi >> 24;

			 hash &= ~hi;

			 But the following is equivalent and a lot
			 faster, especially on modern processors.  */

		      hash ^= hi >> 24;
		    }

		  /* Second part of the modified formula.  This
		     operation can be lifted outside the loop.  */
		  hash &= 0x0fffffff;
		}
	    }
	}
    }
  return hash;
}

#endif /* dl-hash.h */
