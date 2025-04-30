/* Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

/* Test that the compat forwaders in libpthread work correctly.  */

#include <support/test-driver.h>

extern void call_system (void);

int
do_test (void)
{
  /* Calling the system function from a shared library that is not linked
     against libpthread, when the main program is linked against
     libpthread, should not crash.  */
  call_system ();

  return 0;
}

#include <support/test-driver.c>
