/* Copyright (C) 1994-2021 Free Software Foundation, Inc.
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

#include <sys/types.h>
#include <sys/mman.h>
#include <errno.h>

/* Synchronize the region starting at ADDR and extending LEN bytes with the
   file it maps.  Filesystem operations on a file being mapped are
   unpredictable before this is done.  */

int
msync (void *addr, size_t len, int flags)
{
  __set_errno (ENOSYS);
  return -1;
}

stub_warning (msync)
