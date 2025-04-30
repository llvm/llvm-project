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
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


static int fd[4];
static pthread_barrier_t b;
volatile int in_sh_body;
unsigned long cleanups;

static void
cl (void *arg)
{
  cleanups = (cleanups << 4) | (long) arg;
}


static void __attribute__((noinline))
sh_body (void)
{
  char c;

  pthread_cleanup_push (cl, (void *) 1L);

  in_sh_body = 1;
  if (read (fd[2], &c, 1) == 1)
    {
      puts ("read succeeded");
      exit (1);
    }

  pthread_cleanup_pop (0);
}


static void
sh (int sig)
{
  pthread_cleanup_push (cl, (void *) 2L);
  sh_body ();
  in_sh_body = 0;

  pthread_cleanup_pop (0);
}


static void __attribute__((noinline))
tf_body (void)
{
  char c;

  pthread_cleanup_push (cl, (void *) 3L);

  int r = pthread_barrier_wait (&b);
  if (r != 0 && r != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("child thread: barrier_wait failed");
      exit (1);
    }

  if (read (fd[0], &c, 1) == 1)
    {
      puts ("read succeeded");
      exit (1);
    }

  read (fd[0], &c, 1);

  pthread_cleanup_pop (0);
}


static void *
tf (void *arg)
{
  pthread_cleanup_push (cl, (void *) 4L);
  tf_body ();
  pthread_cleanup_pop (0);
  return NULL;
}


static int
do_one_test (void)
{
  in_sh_body = 0;
  cleanups = 0;
  if (pipe (fd) != 0 || pipe (fd + 2) != 0)
    {
      puts ("pipe failed");
      return 1;
    }

  pthread_t th;
  if (pthread_create (&th, NULL, tf, NULL) != 0)
    {
      puts ("create failed");
      return 1;
    }

  int r = pthread_barrier_wait (&b);
  if (r != 0 && r != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("parent thread: barrier_wait failed");
      return 1;
    }

  sleep (1);

  r = pthread_kill (th, SIGHUP);
  if (r)
    {
      errno = r;
      printf ("pthread_kill failed %m\n");
      return 1;
    }

  while (in_sh_body == 0)
    sleep (1);

  if (pthread_cancel (th) != 0)
    {
      puts ("cancel failed");
      return 1;
    }

  void *ret;
  if (pthread_join (th, &ret) != 0)
    {
      puts ("join failed");
      return 1;
    }

  if (ret != PTHREAD_CANCELED)
    {
      puts ("result is wrong");
      return 1;
    }

  if (cleanups != 0x1234L)
    {
      printf ("called cleanups %lx\n", cleanups);
      return 1;
    }

  /* The pipe closing must be issued after the cancellation handling to avoid
     a race condition where the cancellation runs after both pipe ends are
     closed.  In this case the read syscall returns EOF and the cancellation
     must not act.  */
  close (fd[0]);
  close (fd[1]);
  close (fd[2]);
  close (fd[3]);

  return 0;
}


static int
do_test (void)
{
  stack_t ss;
  ss.ss_sp = malloc (2 * SIGSTKSZ);
  if (ss.ss_sp == NULL)
    {
      puts ("failed to allocate alternate stack");
      return 1;
    }
  ss.ss_flags = 0;
  ss.ss_size = 2 * SIGSTKSZ;
  if (sigaltstack (&ss, NULL) < 0)
    {
      printf ("sigaltstack failed %m\n");
      return 1;
    }

  if (pthread_barrier_init (&b, NULL, 2) != 0)
    {
      puts ("barrier_init failed");
      return 1;
    }

  struct sigaction sa;
  sa.sa_handler = sh;
  sigemptyset (&sa.sa_mask);
  sa.sa_flags = 0;

  if (sigaction (SIGHUP, &sa, NULL) != 0)
    {
      puts ("sigaction failed");
      return 1;
    }

  puts ("sa_flags = 0 test");
  if (do_one_test ())
    return 1;

  sa.sa_handler = sh;
  sigemptyset (&sa.sa_mask);
  sa.sa_flags = SA_ONSTACK;

  if (sigaction (SIGHUP, &sa, NULL) != 0)
    {
      puts ("sigaction failed");
      return 1;
    }

  puts ("sa_flags = SA_ONSTACK test");
  if (do_one_test ())
    return 1;

#ifdef SA_SIGINFO
  sa.sa_sigaction = (void (*)(int, siginfo_t *, void *)) sh;
  sigemptyset (&sa.sa_mask);
  sa.sa_flags = SA_SIGINFO;

  if (sigaction (SIGHUP, &sa, NULL) != 0)
    {
      puts ("sigaction failed");
      return 1;
    }

  puts ("sa_flags = SA_SIGINFO test");
  if (do_one_test ())
    return 1;

  sa.sa_sigaction = (void (*)(int, siginfo_t *, void *)) sh;
  sigemptyset (&sa.sa_mask);
  sa.sa_flags = SA_SIGINFO | SA_ONSTACK;

  if (sigaction (SIGHUP, &sa, NULL) != 0)
    {
      puts ("sigaction failed");
      return 1;
    }

  puts ("sa_flags = SA_SIGINFO|SA_ONSTACK test");
  if (do_one_test ())
    return 1;
#endif

  return 0;
}

#define TIMEOUT 40
#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
