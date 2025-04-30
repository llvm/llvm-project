/* BZ #16046 dl_iterate_phdr static executable test.
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
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

#include <link.h>

/* Check that the link map of the static executable itself is iterated
   over exactly once.  */

static int
callback (struct dl_phdr_info *info, size_t size, void *data)
{
  int *count = data;

  if (info->dlpi_name[0] == '\0')
    (*count)++;

  return 0;
}

static int
do_test (void)
{
  int count = 0;
  int status;

  status = dl_iterate_phdr (callback, &count);

  return status || count != 1;
}

#include <support/test-driver.c>
