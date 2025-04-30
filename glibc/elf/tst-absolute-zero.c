/* BZ #23307 absolute zero symbol calculation main executable.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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
#include <support/support.h>
#include <support/test-driver.h>

void *get_absolute (void);

static int
do_test (void)
{
  void *ref = (void *) 0;
  void *ptr;

  ptr = get_absolute ();
  if (ptr != ref)
    FAIL_EXIT1 ("Got %p, expected %p\n", ptr, ref);

  return 0;
}

#include <support/test-driver.c>
