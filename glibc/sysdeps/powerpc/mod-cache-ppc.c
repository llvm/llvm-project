/* Test if an executable can read from rtld_global_ro._dl_cache_line_size.
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

#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <sys/auxv.h>
#include <ldsodefs.h>
#include <errno.h>

/* errnop is required in order to work around BZ #20802.  */
int
test_cache (int *errnop)
{
  int cls1 = GLRO (dl_cache_line_size);
  errno = *errnop;
  uint64_t cls2 = getauxval (AT_DCACHEBSIZE);
  *errnop = errno;

  printf ("AT_DCACHEBSIZE      = %" PRIu64 " B\n", cls2);
  printf ("_dl_cache_line_size = %d B\n", cls1);

  if (cls1 != cls2)
    {
      printf ("error: _dl_cache_line_size != AT_DCACHEBSIZE\n");
      return 1;
    }

  return 0;
}
