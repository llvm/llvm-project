/* mmap with error checking.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#include <support/check.h>
#include <support/xunistd.h>
#include <sys/mman.h>

void *
xmmap (void *addr, size_t length, int prot, int flags, int fd)
{
  void *result = mmap (addr, length, prot, flags, fd, 0);
  if (result == MAP_FAILED)
    FAIL_EXIT1 ("mmap of %zu bytes, prot=0x%x, flags=0x%x: %m",
                length, prot, flags);
  return result;
}
