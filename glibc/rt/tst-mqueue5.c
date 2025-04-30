/* Test mq_notify.
   Copyright (C) 2004-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <fcntl.h>
#include <mqueue.h>
#include <limits.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include "tst-mqueue.h"

#if _POSIX_THREADS && defined SIGRTMIN && defined SA_SIGINFO
# include <pthread.h>

volatile int rtmin_cnt;
volatile pid_t rtmin_pid;
volatile uid_t rtmin_uid;
volatile int rtmin_code;
volatile union sigval rtmin_sigval;

static void
rtmin_handler (int sig, siginfo_t *info, void *ctx)
{
  if (sig != SIGRTMIN)
    abort ();
  ++rtmin_cnt;
  rtmin_pid = info->si_pid;
  rtmin_uid = info->si_uid;
  rtmin_code = info->si_code;
  rtmin_sigval = info->si_value;
}

#define mqsend(q) (mqsend) (q, __LINE__)
static int
(mqsend) (mqd_t q, int line)
{
  char c;
  if (mq_send (q, &c, 1, 1) != 0)
    {
      printf ("mq_send on line %d failed with: %m\n", line);
      return 1;
    }
  return 0;
}

#define mqrecv(q) (mqrecv) (q, __LINE__)
static int
(mqrecv) (mqd_t q, int line)
{
  char c;
  ssize_t rets = TEMP_FAILURE_RETRY (mq_receive (q, &c, 1, NULL));
  if (rets != 1)
    {
      if (rets == -1)
	printf ("mq_receive on line %d failed with: %m\n", line);
      else
	printf ("mq_receive on line %d returned %zd != 1\n",
		line, rets);
      return 1;
    }
  return 0;
}

struct thr_data
{
  const char *name;
  pthread_barrier_t *b3;
  mqd_t q;
};

static void *
thr (void *arg)
{
  pthread_barrier_t *b3 = ((struct thr_data *)arg)->b3;
  mqd_t q = ((struct thr_data *)arg)->q;
  const char *name = ((struct thr_data *)arg)->name;
  int result = 0;

  result |= mqrecv (q);

  (void) pthread_barrier_wait (b3);

  /* Child verifies SIGRTMIN has not been sent.  */

  (void) pthread_barrier_wait (b3);

  /* Parent calls mqsend (q), which should trigger notification.  */

  (void) pthread_barrier_wait (b3);

  if (rtmin_cnt != 2)
    {
      puts ("SIGRTMIN signal in thread did not arrive");
      result = 1;
    }
  else if (rtmin_pid != getppid ()
	   || rtmin_uid != getuid ()
	   || rtmin_code != SI_MESGQ
	   || rtmin_sigval.sival_int != 0xdeadbeef)
    {
      printf ("unexpected siginfo_t fields: pid %u (%u), uid %u (%u), code %d (%d), si_int %d (%d)\n",
	      rtmin_pid, getppid (), rtmin_uid, getuid (),
	      rtmin_code, SI_MESGQ, rtmin_sigval.sival_int, 0xdeadbeef);
      result = 1;
    }

  struct sigevent ev;
  memset (&ev, 0x82, sizeof (ev));
  ev.sigev_notify = SIGEV_NONE;
  if (mq_notify (q, &ev) != 0)
    {
      printf ("mq_notify in thread (q, { SIGEV_NONE }) failed with: %m\n");
      result = 1;
    }

  if (mq_notify (q, NULL) != 0)
    {
      printf ("mq_notify in thread (q, NULL) failed with: %m\n");
      result = 1;
    }

  result |= mqrecv (q);

  (void) pthread_barrier_wait (b3);

  /* Child calls mq_notify (q, { SIGEV_SIGNAL }).  */

  (void) pthread_barrier_wait (b3);

  if (mq_notify (q, NULL) != 0)
    {
      printf ("second mq_notify in thread (q, NULL) failed with: %m\n");
      result = 1;
    }

  (void) pthread_barrier_wait (b3);

  /* Parent calls mqsend (q), which should not trigger notification.  */

  (void) pthread_barrier_wait (b3);

  /* Child verifies SIGRTMIN has not been received.  */
  /* Child calls mq_notify (q, { SIGEV_SIGNAL }).  */

  (void) pthread_barrier_wait (b3);

  mqd_t q4 = mq_open (name, O_RDONLY);
  if (q4 == (mqd_t) -1)
    {
      printf ("mq_open in thread failed with: %m\n");
      result = 1;
    }

  if (mq_notify (q4, NULL) != 0)
    {
      printf ("mq_notify in thread (q4, NULL) failed with: %m\n");
      result = 1;
    }

  if (mq_close (q4) != 0)
    {
      printf ("mq_close in thread failed with: %m\n");
      result = 1;
    }

  (void) pthread_barrier_wait (b3);

  /* Parent calls mqsend (q), which should not trigger notification.  */

  (void) pthread_barrier_wait (b3);

  /* Child verifies SIGRTMIN has not been received.  */
  /* Child calls mq_notify (q, { SIGEV_SIGNAL }).  */

  (void) pthread_barrier_wait (b3);

  mqd_t q5 = mq_open (name, O_WRONLY);
  if (q5 == (mqd_t) -1)
    {
      printf ("mq_open O_WRONLY in thread failed with: %m\n");
      result = 1;
    }

  if (mq_notify (q5, NULL) != 0)
    {
      printf ("mq_notify in thread (q5, NULL) failed with: %m\n");
      result = 1;
    }

  if (mq_close (q5) != 0)
    {
      printf ("mq_close in thread failed with: %m\n");
      result = 1;
    }

  (void) pthread_barrier_wait (b3);

  /* Parent calls mqsend (q), which should not trigger notification.  */

  (void) pthread_barrier_wait (b3);

  /* Child verifies SIGRTMIN has not been received.  */

  return (void *) (long) result;
}

static void
do_child (const char *name, pthread_barrier_t *b2, pthread_barrier_t *b3,
	  mqd_t q)
{
  int result = 0;

  struct sigevent ev;
  memset (&ev, 0x55, sizeof (ev));
  ev.sigev_notify = SIGEV_SIGNAL;
  ev.sigev_signo = SIGRTMIN;
  ev.sigev_value.sival_ptr = &ev;
  if (mq_notify (q, &ev) == 0)
    {
      puts ("first mq_notify in child (q, { SIGEV_SIGNAL }) unexpectedly succeeded");
      result = 1;
    }
  else if (errno != EBUSY)
    {
      printf ("first mq_notify in child (q, { SIGEV_SIGNAL }) failed with: %m\n");
      result = 1;
    }

  (void) pthread_barrier_wait (b2);

  /* Parent calls mqsend (q), which makes notification available.  */

  (void) pthread_barrier_wait (b2);

  rtmin_cnt = 0;

  if (mq_notify (q, &ev) != 0)
    {
      printf ("second mq_notify in child (q, { SIGEV_SIGNAL }) failed with: %m\n");
      result = 1;
    }

  if (rtmin_cnt != 0)
    {
      puts ("SIGRTMIN signal in child caught too early");
      result = 1;
    }

  (void) pthread_barrier_wait (b2);

  /* Parent unsuccessfully attempts to mq_notify.  */
  /* Parent calls mqsend (q), which makes notification available
     and triggers a signal in the child.  */
  /* Parent successfully calls mq_notify SIGEV_SIGNAL.  */

  (void) pthread_barrier_wait (b2);

  if (rtmin_cnt != 1)
    {
      puts ("SIGRTMIN signal in child did not arrive");
      result = 1;
    }
  else if (rtmin_pid != getppid ()
	   || rtmin_uid != getuid ()
	   || rtmin_code != SI_MESGQ
	   || rtmin_sigval.sival_ptr != &ev)
    {
      printf ("unexpected siginfo_t fields: pid %u (%u), uid %u (%u), code %d (%d), si_ptr %p (%p)\n",
	      rtmin_pid, getppid (), rtmin_uid, getuid (),
	      rtmin_code, SI_MESGQ, rtmin_sigval.sival_ptr, &ev);
      result = 1;
    }

  result |= mqsend (q);

  (void) pthread_barrier_wait (b2);

  /* Parent verifies caught SIGRTMIN.  */

  mqd_t q2 = mq_open (name, O_RDWR);
  if (q2 == (mqd_t) -1)
    {
      printf ("mq_open in child failed with: %m\n");
      result = 1;
    }

  (void) pthread_barrier_wait (b2);

  /* Parent mq_open's another mqd_t for the same queue (q3).  */

  memset (&ev, 0x11, sizeof (ev));
  ev.sigev_notify = SIGEV_SIGNAL;
  ev.sigev_signo = SIGRTMIN;
  ev.sigev_value.sival_ptr = &ev;
  if (mq_notify (q2, &ev) != 0)
    {
      printf ("mq_notify in child (q2, { SIGEV_SIGNAL }) failed with: %m\n");
      result = 1;
    }

  (void) pthread_barrier_wait (b2);

  /* Parent unsuccessfully attempts to mq_notify { SIGEV_NONE } on q.  */

  (void) pthread_barrier_wait (b2);

  if (mq_close (q2) != 0)
    {
      printf ("mq_close failed: %m\n");
      result = 1;
    }

  (void) pthread_barrier_wait (b2);

  /* Parent successfully calls mq_notify { SIGEV_NONE } on q3.  */

  (void) pthread_barrier_wait (b2);

  memset (&ev, 0xbb, sizeof (ev));
  ev.sigev_notify = SIGEV_SIGNAL;
  ev.sigev_signo = SIGRTMIN;
  ev.sigev_value.sival_ptr = &b2;
  if (mq_notify (q, &ev) == 0)
    {
      puts ("third mq_notify in child (q, { SIGEV_SIGNAL }) unexpectedly succeeded");
      result = 1;
    }
  else if (errno != EBUSY)
    {
      printf ("third mq_notify in child (q, { SIGEV_SIGNAL }) failed with: %m\n");
      result = 1;
    }

  (void) pthread_barrier_wait (b2);

  /* Parent calls mq_close on q3, which makes the queue available again for
     notification.  */

  (void) pthread_barrier_wait (b2);

  memset (&ev, 0x13, sizeof (ev));
  ev.sigev_notify = SIGEV_NONE;
  if (mq_notify (q, &ev) != 0)
    {
      printf ("mq_notify in child (q, { SIGEV_NONE }) failed with: %m\n");
      result = 1;
    }

  if (mq_notify (q, NULL) != 0)
    {
      printf ("mq_notify in child (q, NULL) failed with: %m\n");
      result = 1;
    }

  (void) pthread_barrier_wait (b2);

  struct thr_data thr_data = { .name = name, .b3 = b3, .q = q };
  pthread_t th;
  int ret = pthread_create (&th, NULL, thr, &thr_data);
  if (ret)
    {
      errno = ret;
      printf ("pthread_created failed with: %m\n");
      result = 1;
    }

  /* Wait till thr calls mq_receive on the empty queue q and blocks on it.  */
  sleep (1);

  memset (&ev, 0x5f, sizeof (ev));
  ev.sigev_notify = SIGEV_SIGNAL;
  ev.sigev_signo = SIGRTMIN;
  ev.sigev_value.sival_int = 0xdeadbeef;
  if (mq_notify (q, &ev) != 0)
    {
      printf ("fourth mq_notify in child (q, { SIGEV_SIGNAL }) failed with: %m\n");
      result = 1;
    }

  /* Ensure the thr thread gets the signal, not us.  */
  sigset_t set;
  sigemptyset (&set);
  sigaddset (&set, SIGRTMIN);
  if (pthread_sigmask (SIG_BLOCK, &set, NULL))
    {
      printf ("Failed to block SIGRTMIN in child: %m\n");
      result = 1;
    }

  (void) pthread_barrier_wait (b2);

  /* Parent calls mqsend (q), which should wake up mqrecv (q)
     in the thread but no notification should be sent.  */

  (void) pthread_barrier_wait (b3);

  if (rtmin_cnt != 1)
    {
      puts ("SIGRTMIN signal caught while thr was blocked on mq_receive");
      result = 1;
    }

  (void) pthread_barrier_wait (b3);

  /* Parent calls mqsend (q), which should trigger notification.  */

  (void) pthread_barrier_wait (b3);

  /* Thread verifies SIGRTMIN has been received.  */
  /* Thread calls mq_notify (q, { SIGEV_NONE }) to verify notification is now
     available for registration.  */
  /* Thread calls mq_notify (q, NULL).  */

  (void) pthread_barrier_wait (b3);

  memset (&ev, 0x6a, sizeof (ev));
  ev.sigev_notify = SIGEV_SIGNAL;
  ev.sigev_signo = SIGRTMIN;
  ev.sigev_value.sival_ptr = do_child;
  if (mq_notify (q, &ev) != 0)
    {
      printf ("fifth mq_notify in child (q, { SIGEV_SIGNAL }) failed with: %m\n");
      result = 1;
    }

  (void) pthread_barrier_wait (b3);

  /* Thread calls mq_notify (q, NULL), which should unregister the above
     notification.  */

  (void) pthread_barrier_wait (b3);

  /* Parent calls mqsend (q), which should not trigger notification.  */

  (void) pthread_barrier_wait (b3);

  if (rtmin_cnt != 2)
    {
      puts ("SIGRTMIN signal caught while notification has been disabled");
      result = 1;
    }

  memset (&ev, 0x7b, sizeof (ev));
  ev.sigev_notify = SIGEV_SIGNAL;
  ev.sigev_signo = SIGRTMIN;
  ev.sigev_value.sival_ptr = thr;
  if (mq_notify (q, &ev) != 0)
    {
      printf ("sixth mq_notify in child (q, { SIGEV_SIGNAL }) failed with: %m\n");
      result = 1;
    }

  (void) pthread_barrier_wait (b3);

  /* Thread opens a new O_RDONLY mqd_t (q4).  */
  /* Thread calls mq_notify (q4, NULL), which should unregister the above
     notification.  */
  /* Thread calls mq_close (q4).  */

  (void) pthread_barrier_wait (b3);

  /* Parent calls mqsend (q), which should not trigger notification.  */

  (void) pthread_barrier_wait (b3);

  if (rtmin_cnt != 2)
    {
      puts ("SIGRTMIN signal caught while notification has been disabled");
      result = 1;
    }

  memset (&ev, 0xe1, sizeof (ev));
  ev.sigev_notify = SIGEV_SIGNAL;
  ev.sigev_signo = SIGRTMIN;
  ev.sigev_value.sival_int = 127;
  if (mq_notify (q, &ev) != 0)
    {
      printf ("seventh mq_notify in child (q, { SIGEV_SIGNAL }) failed with: %m\n");
      result = 1;
    }

  (void) pthread_barrier_wait (b3);

  /* Thread opens a new O_WRONLY mqd_t (q5).  */
  /* Thread calls mq_notify (q5, NULL), which should unregister the above
     notification.  */
  /* Thread calls mq_close (q5).  */

  (void) pthread_barrier_wait (b3);

  /* Parent calls mqsend (q), which should not trigger notification.  */

  (void) pthread_barrier_wait (b3);

  if (rtmin_cnt != 2)
    {
      puts ("SIGRTMIN signal caught while notification has been disabled");
      result = 1;
    }

  /* Reenable test signals before cleaning up the thread.  */
  if (pthread_sigmask (SIG_UNBLOCK, &set, NULL))
    {
      printf ("Failed to unblock SIGRTMIN in child: %m\n");
      result = 1;
    }

  void *thr_ret;
  ret = pthread_join (th, &thr_ret);
  if (ret)
    {
      errno = ret;
      printf ("pthread_join failed: %m\n");
      result = 1;
    }
  else if (thr_ret)
    result = 1;

  if (mq_close (q) != 0)
    {
      printf ("mq_close failed: %m\n");
      result = 1;
    }

  exit (result);
}

#define TEST_FUNCTION do_test ()
static int
do_test (void)
{
  int result = 0;

  char tmpfname[] = "/tmp/tst-mqueue5-barrier.XXXXXX";
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

  pthread_barrier_t *b2;
  b2 = (pthread_barrier_t *) (((uintptr_t) mem + __alignof (pthread_barrier_t))
			      & ~(__alignof (pthread_barrier_t) - 1));

  pthread_barrier_t *b3;
  b3 = b2 + 1;

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

  if (pthread_barrier_init (b2, &a, 2) != 0)
    {
      puts ("barrier_init failed");
      return 1;
    }

  if (pthread_barrier_init (b3, &a, 3) != 0)
    {
      puts ("barrier_init failed");
      return 1;
    }

  if (pthread_barrierattr_destroy (&a) != 0)
    {
      puts ("barrierattr_destroy failed");
      return 1;
    }

  char name[sizeof "/tst-mqueue5-" + sizeof (pid_t) * 3];
  snprintf (name, sizeof (name), "/tst-mqueue5-%u", getpid ());

  struct mq_attr attr = { .mq_maxmsg = 1, .mq_msgsize = 1 };
  mqd_t q = mq_open (name, O_CREAT | O_EXCL | O_RDWR, 0600, &attr);

  if (q == (mqd_t) -1)
    {
      printf ("mq_open failed with: %m\n");
      return result;
    }
  else
    add_temp_mq (name);

  struct sigevent ev;
  memset (&ev, 0xaa, sizeof (ev));
  ev.sigev_notify = SIGEV_NONE;
  if (mq_notify (q, &ev) != 0)
    {
      printf ("mq_notify (q, { SIGEV_NONE }) failed with: %m\n");
      result = 1;
    }

  if (mq_notify (q, &ev) == 0)
    {
      puts ("second mq_notify (q, { SIGEV_NONE }) unexpectedly succeeded");
      result = 1;
    }
  else if (errno != EBUSY)
    {
      printf ("second mq_notify (q, { SIGEV_NONE }) failed with: %m\n");
      result = 1;
    }

  result |= mqsend (q);

  if (mq_notify (q, &ev) != 0)
    {
      printf ("third mq_notify (q, { SIGEV_NONE }) failed with: %m\n");
      result = 1;
    }

  result |= mqrecv (q);

  if (mq_notify (q, NULL) != 0)
    {
      printf ("mq_notify (q, NULL) failed with: %m\n");
      result = 1;
    }

  if (mq_notify (q, NULL) != 0)
    {
      /* Implementation-defined behaviour, so don't fail,
	 just inform.  */
      printf ("second mq_notify (q, NULL) failed with: %m\n");
    }

  struct sigaction sa = { .sa_sigaction = rtmin_handler,
			  .sa_flags = SA_SIGINFO };
  sigemptyset (&sa.sa_mask);
  sigaction (SIGRTMIN, &sa, NULL);

  memset (&ev, 0x55, sizeof (ev));
  ev.sigev_notify = SIGEV_SIGNAL;
  ev.sigev_signo = SIGRTMIN;
  ev.sigev_value.sival_int = 26;
  if (mq_notify (q, &ev) != 0)
    {
      printf ("mq_notify (q, { SIGEV_SIGNAL }) failed with: %m\n");
      result = 1;
    }

  ev.sigev_value.sival_ptr = &ev;
  if (mq_notify (q, &ev) == 0)
    {
      puts ("second mq_notify (q, { SIGEV_SIGNAL }) unexpectedly succeeded");
      result = 1;
    }
  else if (errno != EBUSY)
    {
      printf ("second mq_notify (q, { SIGEV_SIGNAL }) failed with: %m\n");
      result = 1;
    }

  if (rtmin_cnt != 0)
    {
      puts ("SIGRTMIN signal caught too early");
      result = 1;
    }

  result |= mqsend (q);

  if (rtmin_cnt != 1)
    {
      puts ("SIGRTMIN signal did not arrive");
      result = 1;
    }
  else if (rtmin_pid != getpid ()
	   || rtmin_uid != getuid ()
	   || rtmin_code != SI_MESGQ
	   || rtmin_sigval.sival_int != 26)
    {
      printf ("unexpected siginfo_t fields: pid %u (%u), uid %u (%u), code %d (%d), si_int %d (26)\n",
	      rtmin_pid, getpid (), rtmin_uid, getuid (),
	      rtmin_code, SI_MESGQ, rtmin_sigval.sival_int);
      result = 1;
    }

  ev.sigev_value.sival_int = 75;
  if (mq_notify (q, &ev) != 0)
    {
      printf ("third mq_notify (q, { SIGEV_SIGNAL }) failed with: %m\n");
      result = 1;
    }

  result |= mqrecv (q);

  if (mq_notify (q, NULL) != 0)
    {
      printf ("mq_notify (q, NULL) failed with: %m\n");
      result = 1;
    }

  memset (&ev, 0x33, sizeof (ev));
  ev.sigev_notify = SIGEV_NONE;
  if (mq_notify (q, &ev) != 0)
    {
      printf ("fourth mq_notify (q, { SIGEV_NONE }) failed with: %m\n");
      result = 1;
    }

  pid_t pid = fork ();
  if (pid == -1)
    {
      printf ("fork () failed: %m\n");
      mq_unlink (name);
      return 1;
    }

  if (pid == 0)
    do_child (name, b2, b3, q);

  /* Child unsuccessfully attempts to mq_notify.  */

  (void) pthread_barrier_wait (b2);

  result |= mqsend (q);

  (void) pthread_barrier_wait (b2);

  /* Child successfully calls mq_notify SIGEV_SIGNAL now.  */

  result |= mqrecv (q);

  (void) pthread_barrier_wait (b2);

  memset (&ev, 0xbb, sizeof (ev));
  ev.sigev_notify = SIGEV_SIGNAL;
  ev.sigev_signo = SIGRTMIN;
  ev.sigev_value.sival_int = 15;
  if (mq_notify (q, &ev) == 0)
    {
      puts ("fourth mq_notify (q, { SIGEV_SIGNAL }) unexpectedly succeeded");
      result = 1;
    }
  else if (errno != EBUSY)
    {
      printf ("fourth mq_notify (q, { SIGEV_SIGNAL }) failed with: %m\n");
      result = 1;
    }

  result |= mqsend (q);

  if (mq_notify (q, &ev) != 0)
    {
      printf ("fifth mq_notify (q, { SIGEV_SIGNAL }) failed with: %m\n");
      result = 1;
    }

  if (rtmin_cnt != 1)
    {
      puts ("SIGRTMIN signal caught too early");
      result = 1;
    }

  result |= mqrecv (q);

  (void) pthread_barrier_wait (b2);

  /* Child verifies caught SIGRTMIN signal.  */
  /* Child calls mq_send (q) which triggers SIGRTMIN signal here.  */

  (void) pthread_barrier_wait (b2);

  /* Child mq_open's another mqd_t for the same queue (q2).  */

  if (rtmin_cnt != 2)
    {
      puts ("SIGRTMIN signal did not arrive");
      result = 1;
    }
  else if (rtmin_pid != pid
	   || rtmin_uid != getuid ()
	   || rtmin_code != SI_MESGQ
	   || rtmin_sigval.sival_int != 15)
    {
      printf ("unexpected siginfo_t fields: pid %u (%u), uid %u (%u), code %d (%d), si_int %d (15)\n",
	      rtmin_pid, pid, rtmin_uid, getuid (),
	      rtmin_code, SI_MESGQ, rtmin_sigval.sival_int);
      result = 1;
    }

  result |= mqrecv (q);

  (void) pthread_barrier_wait (b2);

  /* Child successfully calls mq_notify { SIGEV_SIGNAL } on q2.  */

  (void) pthread_barrier_wait (b2);

  memset (&ev, 0xbb, sizeof (ev));
  ev.sigev_notify = SIGEV_NONE;
  if (mq_notify (q, &ev) == 0)
    {
      puts ("fifth mq_notify (q, { SIGEV_NONE }) unexpectedly succeeded");
      result = 1;
    }
  else if (errno != EBUSY)
    {
      printf ("fifth mq_notify (q, { SIGEV_NONE }) failed with: %m\n");
      result = 1;
    }

  (void) pthread_barrier_wait (b2);

  /* Child calls mq_close on q2, which makes the queue available again for
     notification.  */

  mqd_t q3 = mq_open (name, O_RDWR);
  if (q3 == (mqd_t) -1)
    {
      printf ("mq_open q3 in parent failed with: %m\n");
      result = 1;
    }

  (void) pthread_barrier_wait (b2);

  memset (&ev, 0x12, sizeof (ev));
  ev.sigev_notify = SIGEV_NONE;
  if (mq_notify (q3, &ev) != 0)
    {
      printf ("mq_notify (q3, { SIGEV_NONE }) failed with: %m\n");
      result = 1;
    }

  (void) pthread_barrier_wait (b2);

  /* Child unsuccessfully attempts to mq_notify { SIGEV_SIGNAL } on q.  */

  (void) pthread_barrier_wait (b2);

  if (mq_close (q3) != 0)
    {
      printf ("mq_close failed: %m\n");
      result = 1;
    }

  (void) pthread_barrier_wait (b2);

  /* Child successfully calls mq_notify { SIGEV_NONE } on q.  */
  /* Child successfully calls mq_notify NULL on q.  */

  (void) pthread_barrier_wait (b2);

  /* Child creates new thread.  */
  /* Thread blocks on mqrecv (q).  */
  /* Child sleeps for 1sec so that thread has time to reach that point.  */
  /* Child successfully calls mq_notify { SIGEV_SIGNAL } on q.  */

  (void) pthread_barrier_wait (b2);

  result |= mqsend (q);

  (void) pthread_barrier_wait (b3);

  /* Child verifies SIGRTMIN has not been sent.  */

  (void) pthread_barrier_wait (b3);

  result |= mqsend (q);

  (void) pthread_barrier_wait (b3);

  /* Thread verifies SIGRTMIN has been caught.  */
  /* Thread calls mq_notify (q, { SIGEV_NONE }) to verify notification is now
     available for registration.  */
  /* Thread calls mq_notify (q, NULL).  */

  (void) pthread_barrier_wait (b3);

  /* Child calls mq_notify (q, { SIGEV_SIGNAL }).  */

  (void) pthread_barrier_wait (b3);

  /* Thread calls mq_notify (q, NULL). */

  (void) pthread_barrier_wait (b3);

  result |= mqsend (q);
  result |= mqrecv (q);

  (void) pthread_barrier_wait (b3);

  /* Child verifies SIGRTMIN has not been sent.  */
  /* Child calls mq_notify (q, { SIGEV_SIGNAL }).  */

  (void) pthread_barrier_wait (b3);

  /* Thread opens a new O_RDONLY mqd_t (q4).  */
  /* Thread calls mq_notify (q4, NULL). */
  /* Thread calls mq_close (q4).  */

  (void) pthread_barrier_wait (b3);

  result |= mqsend (q);
  result |= mqrecv (q);

  (void) pthread_barrier_wait (b3);

  /* Child verifies SIGRTMIN has not been sent.  */
  /* Child calls mq_notify (q, { SIGEV_SIGNAL }).  */

  (void) pthread_barrier_wait (b3);

  /* Thread opens a new O_WRONLY mqd_t (q5).  */
  /* Thread calls mq_notify (q5, NULL). */
  /* Thread calls mq_close (q5).  */

  (void) pthread_barrier_wait (b3);

  result |= mqsend (q);
  result |= mqrecv (q);

  (void) pthread_barrier_wait (b3);

  /* Child verifies SIGRTMIN has not been sent.  */

  int status;
  if (TEMP_FAILURE_RETRY (waitpid (pid, &status, 0)) != pid)
    {
      puts ("waitpid failed");
      kill (pid, SIGKILL);
      result = 1;
    }
  else if (!WIFEXITED (status) || WEXITSTATUS (status))
    {
      printf ("child failed with status %d\n", status);
      result = 1;
    }

  if (mq_unlink (name) != 0)
    {
      printf ("mq_unlink failed: %m\n");
      result = 1;
    }

  if (mq_close (q) != 0)
    {
      printf ("mq_close failed: %m\n");
      result = 1;
    }

  if (mq_notify (q, NULL) == 0)
    {
      puts ("mq_notify on closed mqd_t unexpectedly succeeded");
      result = 1;
    }
  else if (errno != EBADF)
    {
      printf ("mq_notify on closed mqd_t did not fail with EBADF: %m\n");
      result = 1;
    }

  memset (&ev, 0x55, sizeof (ev));
  ev.sigev_notify = SIGEV_NONE;
  if (mq_notify (q, &ev) == 0)
    {
      puts ("mq_notify on closed mqd_t unexpectedly succeeded");
      result = 1;
    }
  else if (errno != EBADF)
    {
      printf ("mq_notify on closed mqd_t did not fail with EBADF: %m\n");
      result = 1;
    }

  return result;
}
#else
# define TEST_FUNCTION 0
#endif

#include "../test-skeleton.c"
