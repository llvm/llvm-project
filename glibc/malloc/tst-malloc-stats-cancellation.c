/* Bug 22830: malloc_stats fails to re-enable cancellation on exit.
   Copyright (C) 2018 Free Software Foundation.
   Copying and distribution of this file, with or without modification,
   are permitted in any medium without royalty provided the copyright
   notice and this notice are preserved. This file is offered as-is,
   without any warranty.  */

#include <errno.h>
#include <stdio.h>
#include <string.h>

#include <pthread.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>

#include <malloc.h>

static void *
test_threadproc (void *gatep)
{
  /* When we are released from the barrier, there is a cancellation
     request pending for this thread.  N.B. pthread_barrier_wait is
     not itself a cancellation point (oddly enough).  */
  pthread_barrier_wait ((pthread_barrier_t *)gatep);
  malloc_stats ();
  fputs ("this call should trigger cancellation\n", stderr);
  return 0;
}

/* We cannot replace stderr with a memstream because writes to memstreams
   do not trigger cancellation.  Instead, main swaps out fd 2 to point to
   a pipe, and this thread reads from the pipe and writes to a memstream
   until EOF, then returns the data accumulated in the memstream.  main
   can't do that itself because, when the test thread gets cancelled,
   it doesn't close the pipe.  */

struct buffer_tp_args
{
  int ifd;
  FILE *real_stderr;
};

static void *
buffer_threadproc (void *argp)
{
  struct buffer_tp_args *args = argp;
  int ifd = args->ifd;
  char block[BUFSIZ], *p;
  ssize_t nread;
  size_t nwritten;

  char *obuf = 0;
  size_t obufsz = 0;
  FILE *ofp = open_memstream (&obuf, &obufsz);
  if (!ofp)
    {
      fprintf (args->real_stderr,
               "buffer_threadproc: open_memstream: %s\n", strerror (errno));
      return 0;
    }

  while ((nread = read (ifd, block, BUFSIZ)) > 0)
    {
      p = block;
      do
        {
          nwritten = fwrite_unlocked (p, 1, nread, ofp);
          if (nwritten == 0)
            {
              fprintf (args->real_stderr,
                       "buffer_threadproc: fwrite_unlocked: %s\n",
                       strerror (errno));
              return 0;
            }
          nread -= nwritten;
          p += nwritten;
        }
      while (nread > 0);
    }
  if (nread == -1)
    {
      fprintf (args->real_stderr, "buffer_threadproc: read: %s\n",
               strerror (errno));
      return 0;
    }
  close (ifd);
  fclose (ofp);
  return obuf;
}


static int
do_test (void)
{
  int result = 0, err, real_stderr_fd, bufpipe[2];
  pthread_t t_thr, b_thr;
  pthread_barrier_t gate;
  void *rv;
  FILE *real_stderr;
  char *obuf;
  void *obuf_v;
  struct buffer_tp_args b_args;

  real_stderr_fd = dup (2);
  if (real_stderr_fd == -1)
    {
      perror ("dup");
      return 2;
    }
  real_stderr = fdopen(real_stderr_fd, "w");
  if (!real_stderr)
    {
      perror ("fdopen");
      return 2;
    }
  if (setvbuf (real_stderr, 0, _IOLBF, 0))
    {
      perror ("setvbuf(real_stderr)");
      return 2;
    }

  if (pipe (bufpipe))
    {
      perror ("pipe");
      return 2;
    }

  /* Below this point, nobody other than the test_threadproc should use
     the normal stderr.  */
  if (dup2 (bufpipe[1], 2) == -1)
    {
      fprintf (real_stderr, "dup2: %s\n", strerror (errno));
      return 2;
    }
  close (bufpipe[1]);

  b_args.ifd = bufpipe[0];
  b_args.real_stderr = real_stderr;
  err = pthread_create (&b_thr, 0, buffer_threadproc, &b_args);
  if (err)
    {
      fprintf (real_stderr, "pthread_create(buffer_thr): %s\n",
               strerror (err));
      return 2;
    }

  err = pthread_barrier_init (&gate, 0, 2);
  if (err)
    {
      fprintf (real_stderr, "pthread_barrier_init: %s\n", strerror (err));
      return 2;
    }

  err = pthread_create (&t_thr, 0, test_threadproc, &gate);
  if (err)
    {
      fprintf (real_stderr, "pthread_create(test_thr): %s\n", strerror (err));
      return 2;
    }

  err = pthread_cancel (t_thr);
  if (err)
    {
      fprintf (real_stderr, "pthread_cancel: %s\n", strerror (err));
      return 2;
    }

  pthread_barrier_wait (&gate); /* cannot fail */

  err = pthread_join (t_thr, &rv);
  if (err)
    {
      fprintf (real_stderr, "pthread_join(test_thr): %s\n", strerror (err));
      return 2;
    }

  /* Closing the normal stderr releases the buffer_threadproc from its
     loop.  */
  fclose (stderr);
  err = pthread_join (b_thr, &obuf_v);
  if (err)
    {
      fprintf (real_stderr, "pthread_join(buffer_thr): %s\n", strerror (err));
      return 2;
    }
  obuf = obuf_v;
  if (obuf == 0)
    return 2; /* error within buffer_threadproc, already reported */

  if (rv != PTHREAD_CANCELED)
    {
      fputs ("FAIL: thread was not cancelled\n", real_stderr);
      result = 1;
    }
  /* obuf should have received all of the text printed by malloc_stats,
     but not the text printed by the final call to fputs.  */
  if (!strstr (obuf, "max mmap bytes"))
    {
      fputs ("FAIL: malloc_stats output incomplete\n", real_stderr);
      result = 1;
    }
  if (strstr (obuf, "this call should trigger cancellation"))
    {
      fputs ("FAIL: fputs produced output\n", real_stderr);
      result = 1;
    }

  if (result == 1)
    {
      fputs ("--- output from thread below ---\n", real_stderr);
      fputs (obuf, real_stderr);
    }
  return result;
}

#include <support/test-driver.c>
