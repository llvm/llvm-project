/* Copyright (C) 2004-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2004.

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

#include <execinfo.h>
#include <pthread.h>
#include <stdio.h>

#define BT_SIZE 64
void *bt_array[BT_SIZE];
int bt_cnt;

int
do_bt (void)
{
  bt_cnt = backtrace (bt_array, BT_SIZE);
  return 56;
}

int
call_do_bt (void)
{
  return do_bt () + 1;
}

void *
tf (void *arg)
{
  if (call_do_bt () != 57)
    return (void *) 1L;
  return NULL;
}

int
do_test (void)
{
  pthread_t th;
  if (pthread_create (&th, NULL, tf, NULL))
    {
      puts ("create failed");
      return 1;
    }

  void *res;
  if (pthread_join (th, &res))
    {
      puts ("join failed");
      return 1;
    }

  if (res != NULL)
    {
      puts ("thread failed");
      return 1;
    }

  char **text = backtrace_symbols (bt_array, bt_cnt);
  if (text == NULL)
    {
      puts ("backtrace_symbols failed");
      return 1;
    }

  for (int i = 0; i < bt_cnt; ++i)
    puts (text[i]);

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
