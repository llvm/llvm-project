/* Support for GNU properties in ldconfig.  Generic version.
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

/* Called for each property in the NT_GNU_PROPERTY_TYPE_0 note of SEGMENT.
   Return value:
     false: Continue processing the properties.
     true : Stop processing the properties.
 */
static inline bool __attribute__ ((always_inline))
read_gnu_property (unsigned int *isal_level, uint32_t type, uint32_t
		   datasz, void *data)
{
  return true;
}

#endif
