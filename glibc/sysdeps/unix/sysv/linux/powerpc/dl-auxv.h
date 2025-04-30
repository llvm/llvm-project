/* Auxiliary vector processing.  Linux/PPC version.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#include <ldsodefs.h>

#if IS_IN (libc) && !defined SHARED
int GLRO(dl_cache_line_size);
#endif

/* Scan the Aux Vector for the "Data Cache Block Size" entry and assign it
   to dl_cache_line_size.  */
#define DL_PLATFORM_AUXV						      \
      case AT_DCACHEBSIZE:						      \
	GLRO(dl_cache_line_size) = av->a_un.a_val;			      \
	break;
