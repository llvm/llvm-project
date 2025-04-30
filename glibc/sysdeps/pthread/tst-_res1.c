/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2003.

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

/* Test whether _res in glibc 2.1.x and earlier (before __res_state()
   was introduced) works.  Portable programs should never do the
   dirty things below.  */

#include <pthread.h>
#include <resolv.h>
#include <stdlib.h>
#include <stdio.h>

void *tf (void *resp)
{
  if (resp == &_res || resp == __res_state ())
    abort ();
  _res.retry = 24;
  return NULL;
}

void do_test (struct __res_state *resp)
{
  if (resp != &_res || resp != __res_state ())
    abort ();
  if (_res.retry != 12)
    abort ();
}

int main (void)
{
#undef _res
  extern struct __res_state _res;
  pthread_t th;

  _res.retry = 12;
  if (pthread_create (&th, NULL, tf, &_res) != 0)
    {
      puts ("create failed");
      exit (1);
    }

  do_test (&_res);

  if (pthread_join (th, NULL) != 0)
    {
      puts ("join failed");
      exit (1);
    }

  do_test (&_res);

  exit (0);
}
