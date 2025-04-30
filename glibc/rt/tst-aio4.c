/* Test for completion signal handling.
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
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <aio.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>

int my_signo;

volatile sig_atomic_t flag;


static void
sighandler (const int signo)
{
  flag = signo;
}

static int
wait_flag (void)
{
  while (flag == 0)
    {
      puts ("Sleeping...");
      sleep (1);
    }

  if (flag != my_signo)
    {
      printf ("signal handler received wrong signal, flag is %d\n", flag);
      return 1;
    }

  return 0;
}

#ifndef SIGRTMIN
# define SIGRTMIN -1
# define SIGRTMAX -1
#endif

static int
do_test (int argc, char *argv[])
{
  char name[] = "/tmp/aio4.XXXXXX";
  int fd;
  struct aiocb *arr[1];
  struct aiocb cb;
  static const char buf[] = "Hello World\n";
  struct aioinit init = {10, 20, 0};
  struct sigaction sa;
  struct sigevent ev;

  if (SIGRTMIN == -1)
  {
      printf ("RT signals not supported.\n");
      return 0;
  }

  /* Select a signal from the middle of the available choices... */
  my_signo = (SIGRTMAX + SIGRTMIN) / 2;

  fd = mkstemp (name);
  if (fd == -1)
    {
      printf ("cannot open temp name: %m\n");
      return 1;
    }

  unlink (name);

  /* Test also aio_init.  */
  aio_init (&init);

  arr[0] = &cb;

  cb.aio_fildes = fd;
  cb.aio_lio_opcode = LIO_WRITE;
  cb.aio_reqprio = 0;
  cb.aio_buf = (void *) buf;
  cb.aio_nbytes = sizeof (buf) - 1;
  cb.aio_offset = 0;
  cb.aio_sigevent.sigev_notify = SIGEV_SIGNAL;
  cb.aio_sigevent.sigev_notify_function = NULL;
  cb.aio_sigevent.sigev_notify_attributes = NULL;
  cb.aio_sigevent.sigev_signo = my_signo;
  cb.aio_sigevent.sigev_value.sival_ptr = NULL;

  ev.sigev_notify = SIGEV_SIGNAL;
  ev.sigev_notify_function = NULL;
  ev.sigev_notify_attributes = NULL;
  ev.sigev_signo = my_signo;

  sa.sa_handler = sighandler;
  sigemptyset (&sa.sa_mask);
  sa.sa_flags = SA_RESTART;

  if (sigaction (my_signo, &sa, NULL) < 0)
    {
      printf ("sigaction failed: %m\n");
      return 1;
    }

  flag = 0;
  /* First use aio_write.  */
  if (aio_write (arr[0]) < 0)
    {
      if (errno == ENOSYS)
	{
	  puts ("no aio support in this configuration");
	  return 0;
	}
      printf ("aio_write failed: %m\n");
      return 1;
    }

  if (wait_flag ())
    return 1;

  puts ("aio_write OK");

  flag = 0;
  /* Again with lio_listio.  */
  if (lio_listio (LIO_NOWAIT, arr, 1, &ev) < 0)
    {
      printf ("lio_listio failed: %m\n");
      return 1;
    }

  if (wait_flag ())
    return 1;

  puts ("all OK");

  return 0;
}

#include "../test-skeleton.c"
