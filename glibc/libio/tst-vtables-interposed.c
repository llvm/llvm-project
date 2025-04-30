/* Test for libio vtables and their validation.  Enabled through interposition.
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

#include "tst-vtables-common.c"

/* Provide an interposed definition of the standard file handles with
   our own vtable.  stdout/stdin/stderr will not work as a result, but
   a succesful test does not print anything, so this is fine.  */
#define _IO_file_jumps jumps
#include "stdfiles.c"

static int
do_test (void)
{
  return run_tests (false);
}

/* Calling setvbuf in the test driver is not supported with our
   interposed file handles.  */
#define TEST_NO_SETVBUF
#include <support/test-driver.c>
