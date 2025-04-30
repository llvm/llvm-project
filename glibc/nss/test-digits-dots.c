/* Copyright (C) 2013-2021 Free Software Foundation, Inc.
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

/* Testcase for BZ #15014 */

#include <stdlib.h>
#include <netdb.h>
#include <errno.h>

#include <support/support.h>

static int
do_test (void)
{
  char buf[32];
  struct hostent *result = NULL;
  struct hostent ret;
  int h_err = 0;
  int err;

  err = gethostbyname_r ("1.2.3.4", &ret, buf, sizeof (buf), &result, &h_err);
  return err == ERANGE && h_err == NETDB_INTERNAL ? EXIT_SUCCESS : EXIT_FAILURE;
}

#include <support/test-driver.c>
