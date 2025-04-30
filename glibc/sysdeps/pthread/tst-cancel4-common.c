/* Common file for all tst-cancel4_*

   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

static int
do_test (void)
{
  if (socketpair (AF_UNIX, SOCK_STREAM, 0, fds) != 0)
    {
      perror ("socketpair");
      exit (1);
    }

  set_socket_buffer (fds[1]);

  if (mktemp (fifoname) == NULL)
    {
      printf ("%s: cannot generate temp file name: %m\n", __func__);
      exit (1);
    }

  int result = 0;
  size_t cnt;
  for (cnt = 0; cnt < ntest_tf; ++cnt)
    {
      if (tests[cnt].only_early)
	continue;

      if (pthread_barrier_init (&b2, NULL, tests[cnt].nb) != 0)
	{
	  puts ("b2 init failed");
	  exit (1);
	}

      /* Reset the counter for the cleanup handler.  */
      cl_called = 0;

      pthread_t th;
      if (pthread_create (&th, NULL, tests[cnt].tf, NULL) != 0)
	{
	  printf ("create for '%s' test failed\n", tests[cnt].name);
	  result = 1;
	  continue;
	}

      int r = pthread_barrier_wait (&b2);
      if (r != 0 && r != PTHREAD_BARRIER_SERIAL_THREAD)
	{
	  printf ("%s: barrier_wait failed\n", __func__);
	  result = 1;
	  continue;
	}

      struct timespec  ts = { .tv_sec = 0, .tv_nsec = 100000000 };
      while (nanosleep (&ts, &ts) != 0)
	continue;

      if (pthread_cancel (th) != 0)
	{
	  printf ("cancel for '%s' failed\n", tests[cnt].name);
	  result = 1;
	  continue;
	}

      void *status;
      if (pthread_join (th, &status) != 0)
	{
	  printf ("join for '%s' failed\n", tests[cnt].name);
	  result = 1;
	  continue;
	}
      if (status != PTHREAD_CANCELED)
	{
	  printf ("thread for '%s' not canceled\n", tests[cnt].name);
	  result = 1;
	  continue;
	}

      if (pthread_barrier_destroy (&b2) != 0)
	{
	  puts ("barrier_destroy failed");
	  result = 1;
	  continue;
	}

      if (cl_called == 0)
	{
	  printf ("cleanup handler not called for '%s'\n", tests[cnt].name);
	  result = 1;
	  continue;
	}
      if (cl_called > 1)
	{
	  printf ("cleanup handler called more than once for '%s'\n",
		  tests[cnt].name);
	  result = 1;
	  continue;
	}

      printf ("in-time cancel test of '%s' successful\n", tests[cnt].name);

      if (tempfd != -1)
	{
	  close (tempfd);
	  tempfd = -1;
	}
      if (tempfd2 != -1)
	{
	  close (tempfd2);
	  tempfd2 = -1;
	}
      if (tempfname != NULL)
	{
	  unlink (tempfname);
	  free (tempfname);
	  tempfname = NULL;
	}
      if (tempmsg != -1)
	{
	  msgctl (tempmsg, IPC_RMID, NULL);
	  tempmsg = -1;
	}
    }

  for (cnt = 0; cnt < ntest_tf; ++cnt)
    {
      if (pthread_barrier_init (&b2, NULL, tests[cnt].nb) != 0)
	{
	  puts ("b2 init failed");
	  exit (1);
	}

      /* Reset the counter for the cleanup handler.  */
      cl_called = 0;

      pthread_t th;
      if (pthread_create (&th, NULL, tests[cnt].tf, (void *) 1l) != 0)
	{
	  printf ("create for '%s' test failed\n", tests[cnt].name);
	  result = 1;
	  continue;
	}

      int r = pthread_barrier_wait (&b2);
      if (r != 0 && r != PTHREAD_BARRIER_SERIAL_THREAD)
	{
	  printf ("%s: barrier_wait failed\n", __func__);
	  result = 1;
	  continue;
	}

      if (pthread_cancel (th) != 0)
	{
	  printf ("cancel for '%s' failed\n", tests[cnt].name);
	  result = 1;
	  continue;
	}

      r = pthread_barrier_wait (&b2);
      if (r != 0 && r != PTHREAD_BARRIER_SERIAL_THREAD)
	{
	  printf ("%s: barrier_wait failed\n", __func__);
	  result = 1;
	  continue;
	}

      void *status;
      if (pthread_join (th, &status) != 0)
	{
	  printf ("join for '%s' failed\n", tests[cnt].name);
	  result = 1;
	  continue;
	}
      if (status != PTHREAD_CANCELED)
	{
	  printf ("thread for '%s' not canceled\n", tests[cnt].name);
	  result = 1;
	  continue;
	}

      if (pthread_barrier_destroy (&b2) != 0)
	{
	  puts ("barrier_destroy failed");
	  result = 1;
	  continue;
	}

      if (cl_called == 0)
	{
	  printf ("cleanup handler not called for '%s'\n", tests[cnt].name);
	  result = 1;
	  continue;
	}
      if (cl_called > 1)
	{
	  printf ("cleanup handler called more than once for '%s'\n",
		  tests[cnt].name);
	  result = 1;
	  continue;
	}

      printf ("early cancel test of '%s' successful\n", tests[cnt].name);

      if (tempfd != -1)
	{
	  close (tempfd);
	  tempfd = -1;
	}
      if (tempfd2 != -1)
	{
	  close (tempfd2);
	  tempfd2 = -1;
	}
      if (tempfname != NULL)
	{
	  unlink (tempfname);
	  free (tempfname);
	  tempfname = NULL;
	}
      if (tempmsg != -1)
	{
	  msgctl (tempmsg, IPC_RMID, NULL);
	  tempmsg = -1;
	}
    }

  return result;
}

#define TIMEOUT 60
#include <support/test-driver.c>
