/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
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

#include <sys/mman.h>
#include <errno.h>

/* Remap arbitrary pages of a shared backing store within an existing
   VMA.  */

int
__remap_file_pages (void *start, size_t size, int prot, size_t pgoff,
		    int flags)
{
  __set_errno (ENOSYS);
  return -1;
}

stub_warning (remap_file_pages)
weak_alias (__remap_file_pages, remap_file_pages)
