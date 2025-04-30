/* Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

#include <unistd.h>
#include <sys/shm.h>
#include <ldsodefs.h>

int
__getshmlba (void)
{
  uint64_t hwcap = GLRO(dl_hwcap);
  int pgsz = GLRO(dl_pagesize);

  if (hwcap & HWCAP_SPARC_V9)
    {
      if (pgsz < (16 * 1024))
	return 16 * 1024;
      else
	return pgsz;
    }
  else if (!(hwcap & HWCAP_SPARC_FLUSH))
    return 64 * 1024;
  else
    return 256 * 1024;
}
