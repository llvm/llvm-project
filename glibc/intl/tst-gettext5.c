/* Test that gettext() in multithreaded applications works correctly if
   different threads operate in different locales referring to the same
   catalog file but with different encodings.
   Copyright (C) 2005-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Bruno Haible <bruno@clisp.org>, 2005.

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

#include <libintl.h>
#include <locale.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Set to 1 if the program is not behaving correctly.  */
int result;

/* Denotes which thread should run next.  */
int flipflop;
/* Lock and wait queue used to switch between the threads.  */
pthread_mutex_t lock;
pthread_cond_t waitqueue;

/* Waits until the flipflop has a given value.
   Before the call, the lock is unlocked.  After the call, it is locked.  */
static void
waitfor (int value)
{
  if (pthread_mutex_lock (&lock))
    exit (10);
  while (flipflop != value)
    if (pthread_cond_wait (&waitqueue, &lock))
      exit (11);
}

/* Sets the flipflop to a given value.
   Before the call, the lock is locked.  After the call, it is unlocked.  */
static void
setto (int value)
{
  flipflop = value;
  if (pthread_cond_signal (&waitqueue))
    exit (20);
  if (pthread_mutex_unlock (&lock))
    exit (21);
}

void *
thread1_execution (void *arg)
{
  char *s;

  waitfor (1);
  uselocale (newlocale (LC_ALL_MASK, "de_DE.ISO-8859-1", NULL));
  setto (2);

  /* Here we expect output in ISO-8859-1.  */

  waitfor (1);
  s = gettext ("cheese");
  puts (s);
  if (strcmp (s, "K\344se"))
    {
      fprintf (stderr, "thread 1 call 1 returned: %s\n", s);
      result = 1;
    }
  setto (2);

  waitfor (1);
  s = gettext ("cheese");
  puts (s);
  if (strcmp (s, "K\344se"))
    {
      fprintf (stderr, "thread 1 call 2 returned: %s\n", s);
      result = 1;
    }
  setto (2);

  return NULL;
}

void *
thread2_execution (void *arg)
{
  char *s;

  waitfor (2);
  uselocale (newlocale (LC_ALL_MASK, "de_DE.UTF-8", NULL));
  setto (1);

  /* Here we expect output in UTF-8.  */

  waitfor (2);
  s = gettext ("cheese");
  puts (s);
  if (strcmp (s, "K\303\244se"))
    {
      fprintf (stderr, "thread 2 call 1 returned: %s\n", s);
      result = 1;
    }
  setto (1);

  waitfor (2);
  s = gettext ("cheese");
  puts (s);
  if (strcmp (s, "K\303\244se"))
    {
      fprintf (stderr, "thread 2 call 2 returned: %s\n", s);
      result = 1;
    }
  setto (1);

  return NULL;
}

int
main (void)
{
  pthread_t thread1;
  pthread_t thread2;

  unsetenv ("LANGUAGE");
  unsetenv ("OUTPUT_CHARSET");
  textdomain ("codeset");
  bindtextdomain ("codeset", OBJPFX "domaindir");
  result = 0;

  flipflop = 1;
  if (pthread_mutex_init (&lock, NULL))
    exit (2);
  if (pthread_cond_init (&waitqueue, NULL))
    exit (2);
  if (pthread_create (&thread1, NULL, &thread1_execution, NULL))
    exit (2);
  if (pthread_create (&thread2, NULL, &thread2_execution, NULL))
    exit (2);
  if (pthread_join (thread2, NULL))
    exit (3);

  return result;
}
