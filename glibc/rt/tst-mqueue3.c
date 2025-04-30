/* Test SIGEV_THREAD handling for POSIX message queues.
   Copyright (C) 2004-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2004.

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
#include <mqueue.h>
#include <signal.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#if _POSIX_THREADS
# include <pthread.h>

static pid_t pid;
static mqd_t m;
static const char message[] = "hello";

# define MAXMSG 10
# define MSGSIZE 10
# define UNIQUE 42


static void
fct (union sigval s)
{
  /* Put the mq in non-blocking mode.  */
  struct mq_attr attr;
  if (mq_getattr (m, &attr) != 0)
    {
      printf ("%s: mq_getattr failed: %m\n", __FUNCTION__);
      exit (1);
    }
  attr.mq_flags |= O_NONBLOCK;
  if (mq_setattr (m, &attr, NULL) != 0)
    {
      printf ("%s: mq_setattr failed: %m\n", __FUNCTION__);
      exit (1);
    }

  /* Check the values.  */
  if (attr.mq_maxmsg != MAXMSG)
    {
      printf ("%s: mq_maxmsg wrong: is %jd, expecte %d\n",
	      __FUNCTION__, (intmax_t) attr.mq_maxmsg, MAXMSG);
      exit (1);
    }
  if (attr.mq_msgsize != MAXMSG)
    {
      printf ("%s: mq_msgsize wrong: is %jd, expecte %d\n",
	      __FUNCTION__, (intmax_t) attr.mq_msgsize, MSGSIZE);
      exit (1);
    }

  /* Read the message.  */
  char buf[attr.mq_msgsize];
  ssize_t n = TEMP_FAILURE_RETRY (mq_receive (m, buf, attr.mq_msgsize, NULL));
  if (n != sizeof (message))
    {
      printf ("%s: length of message wrong: is %zd, expected %zu\n",
	      __FUNCTION__, n, sizeof (message));
      exit (1);
    }
  if (memcmp (buf, message, sizeof (message)) != 0)
    {
      printf ("%s: message wrong: is \"%s\", expected \"%s\"\n",
	      __FUNCTION__, buf, message);
      exit (1);
    }

  exit (UNIQUE);
}


int
do_test (void)
{
  char tmpfname[] = "/tmp/tst-mqueue3-barrier.XXXXXX";
  int fd = mkstemp (tmpfname);
  if (fd == -1)
    {
      printf ("cannot open temporary file: %m\n");
      return 1;
    }

  /* Make sure it is always removed.  */
  unlink (tmpfname);

  /* Create one page of data.  */
  size_t ps = sysconf (_SC_PAGESIZE);
  char data[ps];
  memset (data, '\0', ps);

  /* Write the data to the file.  */
  if (write (fd, data, ps) != (ssize_t) ps)
    {
      puts ("short write");
      return 1;
    }

  void *mem = mmap (NULL, ps, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (mem == MAP_FAILED)
    {
      printf ("mmap failed: %m\n");
      return 1;
    }

  pthread_barrier_t *b;
  b = (pthread_barrier_t *) (((uintptr_t) mem + __alignof (pthread_barrier_t))
                             & ~(__alignof (pthread_barrier_t) - 1));

  pthread_barrierattr_t a;
  if (pthread_barrierattr_init (&a) != 0)
    {
      puts ("barrierattr_init failed");
      return 1;
    }

  if (pthread_barrierattr_setpshared (&a, PTHREAD_PROCESS_SHARED) != 0)
    {
      puts ("barrierattr_setpshared failed, could not test");
      return 0;
    }

  if (pthread_barrier_init (b, &a, 2) != 0)
    {
      puts ("barrier_init failed");
      return 1;
    }

  if (pthread_barrierattr_destroy (&a) != 0)
    {
      puts ("barrierattr_destroy failed");
      return 1;
    }

  /* Name for the message queue.  */
  char mqname[sizeof ("/tst-mqueue3-") + 3 * sizeof (pid_t)];
  snprintf (mqname, sizeof (mqname) - 1, "/tst-mqueue3-%ld",
	    (long int) getpid ());

  /* Create the message queue.  */
  struct mq_attr attr = { .mq_maxmsg = MAXMSG, .mq_msgsize = MSGSIZE };
  m = mq_open (mqname, O_CREAT | O_EXCL | O_RDWR, 0600, &attr);
  if (m == -1)
    {
      if (errno == ENOSYS)
	{
	  puts ("not implemented");
	  return 0;
	}

      puts ("mq_open failed");
      return 1;
    }

  /* Unlink the message queue right away.  */
  if (mq_unlink (mqname) != 0)
    {
      puts ("mq_unlink failed");
      return 1;
    }

  pid = fork ();
  if (pid == -1)
    {
      puts ("fork failed");
      return 1;
    }
  if (pid == 0)
    {
      /* Request notification via thread.  */
      struct sigevent ev;
      ev.sigev_notify = SIGEV_THREAD;
      ev.sigev_notify_function = fct;
      ev.sigev_value.sival_ptr = NULL;
      ev.sigev_notify_attributes = NULL;

      /* Tell the kernel.  */
      if (mq_notify (m,&ev) != 0)
	{
	  puts ("mq_notify failed");
	  exit (1);
	}

      /* Tell the parent we are ready.  */
      (void) pthread_barrier_wait (b);

      /* Make sure the process goes away eventually.  */
      alarm (10);

      /* Do nothing forever.  */
      while (1)
	pause ();
    }

  /* Wait for the child process to register to notification method.  */
  (void) pthread_barrier_wait (b);

  /* Send the message.  */
  if (mq_send (m, message, sizeof (message), 1) != 0)
    {
      kill (pid, SIGKILL);
      puts ("mq_send failed");
      return 1;
    }

  int r;
  if (TEMP_FAILURE_RETRY (waitpid (pid, &r, 0)) != pid)
    {
      kill (pid, SIGKILL);
      puts ("waitpid failed");
      return 1;
    }

  return WIFEXITED (r) && WEXITSTATUS (r) == UNIQUE ? 0 : 1;
}
# define TEST_FUNCTION do_test ()
#else
# define TEST_FUNCTION 0
#endif

#include "../test-skeleton.c"
