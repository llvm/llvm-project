/* Run-time dynamic linker data structures for loaded ELF shared objects.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
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

#ifndef	_LDSODEFS_H

/* We have the auxiliary vector.  */
#define HAVE_AUX_VECTOR

/* Get the real definitions.  */
#include_next <ldsodefs.h>

/* We can assume that the kernel always provides the AT_UID, AT_EUID,
   AT_GID, and AT_EGID values in the auxiliary vector from 2.4.0 or so on.  */
#define HAVE_AUX_XID

/* We can assume that the kernel always provides the AT_SECURE value
   in the auxiliary vector from 2.5.74 or so on.  */
#define HAVE_AUX_SECURE

/* Starting with one of the 2.4.0 pre-releases the Linux kernel passes
   up the page size information.  */
#define HAVE_AUX_PAGESIZE

#endif /* ldsodefs.h */
