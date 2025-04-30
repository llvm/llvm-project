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

#include <errno.h>
#include <error.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>

volatile int count;

void
sh (int sig)
{
  ++count;
}

int
main (void)
{
  struct sigaction sa;
  sa.sa_handler = sh;
  sigemptyset (&sa.sa_mask);
  sa.sa_flags = 0;
  if (sigaction (SIGUSR1, &sa, NULL) < 0)
    {
      printf ("sigaction failed: %m\n");
      exit (1);
    }
  if (raise (SIGUSR1) < 0)
    {
      printf ("first raise failed: %m\n");
      exit (1);
    }
  if (raise (SIGUSR1) < 0)
    {
      printf ("second raise failed: %m\n");
      exit (1);
    }
  if (count != 2)
    {
      printf ("signal handler not called 2 times\n");
      exit (1);
    }
  exit (0);
}
