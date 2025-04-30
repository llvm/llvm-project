/* Support for GNU properties in ldconfig.  x86 version.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

#ifndef _ELF_READ_PROP_H
#define _ELF_READ_PROP_H

#include <dl-cache.h>

/* Called for each property in the NT_GNU_PROPERTY_TYPE_0 note of SEGMENT.
   Return value:
     false: Continue processing the properties.
     true : Stop processing the properties.
 */
static inline bool __attribute__ ((always_inline))
read_gnu_property (unsigned int *isal_level, uint32_t type,
		   uint32_t datasz, void *data)
{
  /* Property type must be in ascending order.  */
  if (type > GNU_PROPERTY_X86_ISA_1_NEEDED)
    return true;

  if (type == GNU_PROPERTY_X86_ISA_1_NEEDED)
    {
      if (datasz == 4)
	{
	  /* The size of GNU_PROPERTY_X86_ISA_1_NEEDED must be 4 bytes.
	     There is no point to continue if this type is ill-formed.  */
	  unsigned int isa_1_needed = *(unsigned int *) data;
	  _Static_assert (((sizeof (isa_1_needed) * 8)
			   <= (1 << DL_CACHE_HWCAP_ISA_LEVEL_COUNT)),
			  "DL_CACHE_HWCAP_ISA_LEVEL_COUNT is too small");
	  if (isa_1_needed != 0)
	    {
	      unsigned int level;
	      asm ("bsr %1, %0" : "=r" (level) : "g" (isa_1_needed));
	      *isal_level = level;
	    }
	}
      return true;
    }

  return false;
}

#endif
