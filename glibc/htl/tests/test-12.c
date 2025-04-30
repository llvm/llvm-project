/* Test concurrency level.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library;  if not, see
   <https://www.gnu.org/licenses/>.  */

#define _GNU_SOURCE

#include <pthread.h>
#include <assert.h>
#include <error.h>
#include <errno.h>

int
main (int argc, char **argv)
{
  int i;
  int err;

  i = pthread_getconcurrency ();
  assert (i == 0);

  err = pthread_setconcurrency (-1);
  assert (err == EINVAL);

  err = pthread_setconcurrency (4);
  assert (err == 0);

  i = pthread_getconcurrency ();
  assert (i == 4);

  return 0;
}
