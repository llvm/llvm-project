/* Optional code to distinguish library flavours.  x86-64 version.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#ifndef _DL_LIBRECON_H

#include <sysdeps/unix/sysv/linux/dl-librecon.h>

/* Recognizing extra environment variables.  For 64-bit applications,
   branch prediction performance may be negatively impacted when the
   target of a branch is more than 4GB away from the branch.  Add the
   Prefer_MAP_32BIT_EXEC bit so that mmap will try to map executable
   pages with MAP_32BIT first.  NB: MAP_32BIT will map to lower 2GB,
   not lower 4GB, address.  Prefer_MAP_32BIT_EXEC reduces bits available
   for address space layout randomization (ASLR).  Prefer_MAP_32BIT_EXEC
   is always disabled for SUID programs and can be enabled by setting
   environment variable, LD_PREFER_MAP_32BIT_EXEC.  */
#define EXTRA_LD_ENVVARS \
  case 21:								  \
    if (!__libc_enable_secure						  \
	&& memcmp (envline, "PREFER_MAP_32BIT_EXEC", 21) == 0)		  \
      GLRO(dl_x86_cpu_features).preferred[index_arch_Prefer_MAP_32BIT_EXEC] \
	|= bit_arch_Prefer_MAP_32BIT_EXEC;				  \
    break;

/* Extra unsecure variables.  The names are all stuffed in a single
   string which means they have to be terminated with a '\0' explicitly.  */
#define EXTRA_UNSECURE_ENVVARS \
  "LD_PREFER_MAP_32BIT_EXEC\0"

#endif /* dl-librecon.h */
