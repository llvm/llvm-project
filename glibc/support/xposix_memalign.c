/* Error-checking wrapper for posix_memalign.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

#include <support/support.h>
#include <stdlib.h>
#include <errno.h>

void *
xposix_memalign (size_t alignment, size_t n)
{
  void *p = NULL;

  int ret = posix_memalign (&p, alignment, n);
  if (ret)
    {
      errno = ret;
      oom_error ("posix_memalign", n);
    }
  return p;
}
