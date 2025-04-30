/* Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2002.

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

#define _GNU_SOURCE	1
#include <argp.h>
#include <error.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/param.h>
#include <sys/types.h>

#ifndef MAX_THREADS
# define MAX_THREADS		100000
#endif
#ifndef DEFAULT_THREADS
# define DEFAULT_THREADS	50
#endif


#define OPT_TO_THREAD		300
#define OPT_TO_PROCESS		301
#define OPT_SYNC_SIGNAL		302
#define OPT_SYNC_JOIN		303
#define OPT_TOPLEVEL		304


static const struct argp_option options[] =
  {
    { NULL, 0, NULL, 0, "\
This is a test for threads so we allow ther user to selection the number of \
threads which are used at any one time.  Independently the total number of \
rounds can be selected.  This is the total number of threads which will have \
run when the process terminates:" },
    { "threads", 't', "NUMBER", 0, "Number of threads used at once" },
    { "starts", 's', "NUMBER", 0, "Total number of working threads" },
    { "toplevel", OPT_TOPLEVEL, "NUMBER", 0,
      "Number of toplevel threads which start the other threads; this \
implies --sync-join" },

    { NULL, 0, NULL, 0, "\
Each thread can do one of two things: sleep or do work.  The latter is 100% \
CPU bound.  The work load is the probability a thread does work.  All values \
from zero to 100 (inclusive) are valid.  How often each thread repeats this \
can be determined by the number of rounds.  The work cost determines how long \
each work session (not sleeping) takes.  If it is zero a thread would \
effectively nothing.  By setting the number of rounds to zero the thread \
does no work at all and pure thread creation times can be measured." },
    { "workload", 'w', "PERCENT", 0, "Percentage of time spent working" },
    { "workcost", 'c', "NUMBER", 0,
      "Factor in the cost of each round of working" },
    { "rounds", 'r', "NUMBER", 0, "Number of rounds each thread runs" },

    { NULL, 0, NULL, 0, "\
There are a number of different methods how thread creation can be \
synchronized.  Synchronization is necessary since the number of concurrently \
running threads is limited." },
    { "sync-signal", OPT_SYNC_SIGNAL, NULL, 0,
      "Synchronize using a signal (default)" },
    { "sync-join", OPT_SYNC_JOIN, NULL, 0, "Synchronize using pthread_join" },

    { NULL, 0, NULL, 0, "\
One parameter for each threads execution is the size of the stack.  If this \
parameter is not used the system's default stack size is used.  If many \
threads are used the stack size should be chosen quite small." },
    { "stacksize", 'S', "BYTES", 0, "Size of threads stack" },
    { "guardsize", 'g', "BYTES", 0,
      "Size of stack guard area; must fit into the stack" },

    { NULL, 0, NULL, 0, "Signal options:" },
    { "to-thread", OPT_TO_THREAD, NULL, 0, "Send signal to main thread" },
    { "to-process", OPT_TO_PROCESS, NULL, 0,
      "Send signal to process (default)" },

    { NULL, 0, NULL, 0, "Administrative options:" },
    { "progress", 'p', NULL, 0, "Show signs of progress" },
    { "timing", 'T', NULL, 0,
      "Measure time from startup to the last thread finishing" },
    { NULL, 0, NULL, 0, NULL }
  };

/* Prototype for option handler.  */
static error_t parse_opt (int key, char *arg, struct argp_state *state);

/* Data structure to communicate with argp functions.  */
static struct argp argp =
{
  options, parse_opt
};


static unsigned long int threads = DEFAULT_THREADS;
static unsigned long int workload = 75;
static unsigned long int workcost = 20;
static unsigned long int rounds = 10;
static long int starts = 5000;
static unsigned long int stacksize;
static long int guardsize = -1;
static bool progress;
static bool timing;
static bool to_thread;
static unsigned long int toplevel = 1;


static long int running;
static pthread_mutex_t running_mutex = PTHREAD_MUTEX_INITIALIZER;

static pid_t pid;
static pthread_t tmain;

static clockid_t cl;
static struct timespec start_time;


static pthread_mutex_t sum_mutex = PTHREAD_MUTEX_INITIALIZER;
unsigned int sum;

static enum
  {
    sync_signal,
    sync_join
  }
sync_method;


/* We use 64bit values for the times.  */
typedef unsigned long long int hp_timing_t;


/* Attributes for all created threads.  */
static pthread_attr_t attr;


static void *
work (void *arg)
{
  unsigned long int i;
  unsigned int state = (unsigned long int) arg;

  for (i = 0; i < rounds; ++i)
    {
      /* Determine what to do.  */
      unsigned int rnum;

      /* Uniform distribution.  */
      do
	rnum = rand_r (&state);
      while (rnum >= UINT_MAX - (UINT_MAX % 100));

      rnum %= 100;

      if (rnum < workload)
	{
	  int j;
	  int a[4] = { i, rnum, i + rnum, rnum - i };

	  if (progress)
	    write (STDERR_FILENO, "c", 1);

	  for (j = 0; j < workcost; ++j)
	    {
	      a[0] += a[3] >> 12;
	      a[1] += a[2] >> 20;
	      a[2] += a[1] ^ 0x3423423;
	      a[3] += a[0] - a[1];
	    }

	  pthread_mutex_lock (&sum_mutex);
	  sum += a[0] + a[1] + a[2] + a[3];
	  pthread_mutex_unlock (&sum_mutex);
	}
      else
	{
	  /* Just sleep.  */
	  struct timespec tv;

	  tv.tv_sec = 0;
	  tv.tv_nsec = 10000000;

	  if (progress)
	    write (STDERR_FILENO, "w", 1);

	  nanosleep (&tv, NULL);
	}
    }

  return NULL;
}


static void *
thread_function (void *arg)
{
  work (arg);

  pthread_mutex_lock (&running_mutex);
  if (--running <= 0 && starts <= 0)
    {
      /* We are done.  */
      if (progress)
	write (STDERR_FILENO, "\n", 1);

      if (timing)
	{
	  struct timespec end_time;

	  if (clock_gettime (cl, &end_time) == 0)
	    {
	      end_time.tv_sec -= start_time.tv_sec;
	      end_time.tv_nsec -= start_time.tv_nsec;
	      if (end_time.tv_nsec < 0)
		{
		  end_time.tv_nsec += 1000000000;
		  --end_time.tv_sec;
		}

	      printf ("\nRuntime: %lu.%09lu seconds\n",
		      (unsigned long int) end_time.tv_sec,
		      (unsigned long int) end_time.tv_nsec);
	    }
	}

      printf ("Result: %08x\n", sum);

      exit (0);
    }
  pthread_mutex_unlock (&running_mutex);

  if (sync_method == sync_signal)
    {
      if (to_thread)
	/* This code sends a signal to the main thread.  */
	pthread_kill (tmain, SIGUSR1);
      else
	/* Use this code to test sending a signal to the process.  */
	kill (pid, SIGUSR1);
    }

  if (progress)
    write (STDERR_FILENO, "f", 1);

  return NULL;
}


struct start_info
{
  unsigned int starts;
  unsigned int threads;
};


static void *
start_threads (void *arg)
{
  struct start_info *si = arg;
  unsigned int starts = si->starts;
  pthread_t ths[si->threads];
  unsigned int state = starts;
  unsigned int n;
  unsigned int i = 0;
  int err;

  if (progress)
    write (STDERR_FILENO, "T", 1);

  memset (ths, '\0', sizeof (pthread_t) * si->threads);

  while (starts-- > 0)
    {
      if (ths[i] != 0)
	{
	  /* Wait for the threads in the order they were created.  */
	  err = pthread_join (ths[i], NULL);
	  if (err != 0)
	    error (EXIT_FAILURE, err, "cannot join thread");

	  if (progress)
	    write (STDERR_FILENO, "f", 1);
	}

      err = pthread_create (&ths[i], &attr, work,
			    (void *) (long) (rand_r (&state) + starts + i));

      if (err != 0)
	error (EXIT_FAILURE, err, "cannot start thread");

      if (progress)
	write (STDERR_FILENO, "t", 1);

      if (++i == si->threads)
	i = 0;
    }

  n = i;
  do
    {
      if (ths[i] != 0)
	{
	  err = pthread_join (ths[i], NULL);
	  if (err != 0)
	    error (EXIT_FAILURE, err, "cannot join thread");

	  if (progress)
	    write (STDERR_FILENO, "f", 1);
	}

      if (++i == si->threads)
	i = 0;
    }
  while (i != n);

  if (progress)
    write (STDERR_FILENO, "F", 1);

  return NULL;
}


int
main (int argc, char *argv[])
{
  int remaining;
  sigset_t ss;
  pthread_t th;
  pthread_t *ths = NULL;
  int empty = 0;
  int last;
  bool cont = true;

  /* Parse and process arguments.  */
  argp_parse (&argp, argc, argv, 0, &remaining, NULL);

  if (sync_method == sync_join)
    {
      ths = (pthread_t *) calloc (threads, sizeof (pthread_t));
      if (ths == NULL)
	error (EXIT_FAILURE, errno,
	       "cannot allocate memory for thread descriptor array");

      last = threads;
    }
  else
    {
      ths = &th;
      last = 1;
    }

  if (toplevel > threads)
    {
      printf ("resetting number of toplevel threads to %lu to not surpass number to concurrent threads\n",
	      threads);
      toplevel = threads;
    }

  if (timing)
    {
      if (clock_getcpuclockid (0, &cl) != 0
	  || clock_gettime (cl, &start_time) != 0)
	timing = false;
    }

  /* We need this later.  */
  pid = getpid ();
  tmain = pthread_self ();

  /* We use signal SIGUSR1 for communication between the threads and
     the main thread.  We only want sychronous notification.  */
  if (sync_method == sync_signal)
    {
      sigemptyset (&ss);
      sigaddset (&ss, SIGUSR1);
      if (sigprocmask (SIG_BLOCK, &ss, NULL) != 0)
	error (EXIT_FAILURE, errno, "cannot set signal mask");
    }

  /* Create the thread attributes.  */
  pthread_attr_init (&attr);

  /* If the user provided a stack size use it.  */
  if (stacksize != 0
      && pthread_attr_setstacksize (&attr, stacksize) != 0)
    puts ("could not set stack size; will use default");
  /* And stack guard size.  */
  if (guardsize != -1
      && pthread_attr_setguardsize (&attr, guardsize) != 0)
    puts ("invalid stack guard size; will use default");

  /* All threads are created detached if we are not using pthread_join
     to synchronize.  */
  if (sync_method != sync_join)
    pthread_attr_setdetachstate (&attr, PTHREAD_CREATE_DETACHED);

  if (sync_method == sync_signal)
    {
      while (1)
	{
	  int err;
	  bool do_wait = false;

	  pthread_mutex_lock (&running_mutex);
	  if (starts-- < 0)
	    cont = false;
	  else
	    do_wait = ++running >= threads && starts > 0;

	  pthread_mutex_unlock (&running_mutex);

	  if (! cont)
	    break;

	  if (progress)
	    write (STDERR_FILENO, "t", 1);

	  err = pthread_create (&ths[empty], &attr, thread_function,
				(void *) starts);
	  if (err != 0)
	    error (EXIT_FAILURE, err, "cannot start thread %lu", starts);

	  if (++empty == last)
	    empty = 0;

	  if (do_wait)
	    sigwaitinfo (&ss, NULL);
	}

      /* Do nothing anymore.  On of the threads will terminate the program.  */
      sigfillset (&ss);
      sigdelset (&ss, SIGINT);
      while (1)
	sigsuspend (&ss);
    }
  else
    {
      pthread_t ths[toplevel];
      struct start_info si[toplevel];
      unsigned int i;

      for (i = 0; i < toplevel; ++i)
	{
	  unsigned int child_starts = starts / (toplevel - i);
	  unsigned int child_threads = threads / (toplevel - i);
	  int err;

	  si[i].starts = child_starts;
	  si[i].threads = child_threads;

	  err = pthread_create (&ths[i], &attr, start_threads, &si[i]);
	  if (err != 0)
	    error (EXIT_FAILURE, err, "cannot start thread");

	  starts -= child_starts;
	  threads -= child_threads;
	}

      for (i = 0; i < toplevel; ++i)
	{
	  int err = pthread_join (ths[i], NULL);

	  if (err != 0)
	    error (EXIT_FAILURE, err, "cannot join thread");
	}

      /* We are done.  */
      if (progress)
	write (STDERR_FILENO, "\n", 1);

      if (timing)
	{
	  struct timespec end_time;

	  if (clock_gettime (cl, &end_time) == 0)
	    {
	      end_time.tv_sec -= start_time.tv_sec;
	      end_time.tv_nsec -= start_time.tv_nsec;
	      if (end_time.tv_nsec < 0)
		{
		  end_time.tv_nsec += 1000000000;
		  --end_time.tv_sec;
		}

	      printf ("\nRuntime: %lu.%09lu seconds\n",
		      (unsigned long int) end_time.tv_sec,
		      (unsigned long int) end_time.tv_nsec);
	    }
	}

      printf ("Result: %08x\n", sum);

      exit (0);
    }

  /* NOTREACHED */
  return 0;
}


/* Handle program arguments.  */
static error_t
parse_opt (int key, char *arg, struct argp_state *state)
{
  unsigned long int num;
  long int snum;

  switch (key)
    {
    case 't':
      num = strtoul (arg, NULL, 0);
      if (num <= MAX_THREADS)
	threads = num;
      else
	printf ("\
number of threads limited to %u; recompile with a higher limit if necessary",
		MAX_THREADS);
      break;

    case 'w':
      num = strtoul (arg, NULL, 0);
      if (num <= 100)
	workload = num;
      else
	puts ("workload must be between 0 and 100 percent");
      break;

    case 'c':
      workcost = strtoul (arg, NULL, 0);
      break;

    case 'r':
      rounds = strtoul (arg, NULL, 0);
      break;

    case 's':
      starts = strtoul (arg, NULL, 0);
      break;

    case 'S':
      num = strtoul (arg, NULL, 0);
      if (num >= PTHREAD_STACK_MIN)
	stacksize = num;
      else
	printf ("minimum stack size is %d\n", PTHREAD_STACK_MIN);
      break;

    case 'g':
      snum = strtol (arg, NULL, 0);
      if (snum < 0)
	printf ("invalid guard size %s\n", arg);
      else
	guardsize = snum;
      break;

    case 'p':
      progress = true;
      break;

    case 'T':
      timing = true;
      break;

    case OPT_TO_THREAD:
      to_thread = true;
      break;

    case OPT_TO_PROCESS:
      to_thread = false;
      break;

    case OPT_SYNC_SIGNAL:
      sync_method = sync_signal;
      break;

    case OPT_SYNC_JOIN:
      sync_method = sync_join;
      break;

    case OPT_TOPLEVEL:
      num = strtoul (arg, NULL, 0);
      if (num < MAX_THREADS)
	toplevel = num;
      else
	printf ("\
number of threads limited to %u; recompile with a higher limit if necessary",
		MAX_THREADS);
      sync_method = sync_join;
      break;

    default:
      return ARGP_ERR_UNKNOWN;
    }

  return 0;
}


static hp_timing_t
get_clockfreq (void)
{
  /* We read the information from the /proc filesystem.  It contains at
     least one line like
	cpu MHz         : 497.840237
     or also
	cpu MHz         : 497.841
     We search for this line and convert the number in an integer.  */
  static hp_timing_t result;
  int fd;

  /* If this function was called before, we know the result.  */
  if (result != 0)
    return result;

  fd = open ("/proc/cpuinfo", O_RDONLY);
  if (__glibc_likely (fd != -1))
    {
      /* XXX AFAIK the /proc filesystem can generate "files" only up
         to a size of 4096 bytes.  */
      char buf[4096];
      ssize_t n;

      n = read (fd, buf, sizeof buf);
      if (__builtin_expect (n, 1) > 0)
	{
	  char *mhz = memmem (buf, n, "cpu MHz", 7);

	  if (__glibc_likely (mhz != NULL))
	    {
	      char *endp = buf + n;
	      int seen_decpoint = 0;
	      int ndigits = 0;

	      /* Search for the beginning of the string.  */
	      while (mhz < endp && (*mhz < '0' || *mhz > '9') && *mhz != '\n')
		++mhz;

	      while (mhz < endp && *mhz != '\n')
		{
		  if (*mhz >= '0' && *mhz <= '9')
		    {
		      result *= 10;
		      result += *mhz - '0';
		      if (seen_decpoint)
			++ndigits;
		    }
		  else if (*mhz == '.')
		    seen_decpoint = 1;

		  ++mhz;
		}

	      /* Compensate for missing digits at the end.  */
	      while (ndigits++ < 6)
		result *= 10;
	    }
	}

      close (fd);
    }

  return result;
}


int
clock_getcpuclockid (pid_t pid, clockid_t *clock_id)
{
  /* We don't allow any process ID but our own.  */
  if (pid != 0 && pid != getpid ())
    return EPERM;

#ifdef CLOCK_PROCESS_CPUTIME_ID
  /* Store the number.  */
  *clock_id = CLOCK_PROCESS_CPUTIME_ID;

  return 0;
#else
  /* We don't have a timer for that.  */
  return ENOENT;
#endif
}


#ifdef i386
#define HP_TIMING_NOW(Var)	__asm__ __volatile__ ("rdtsc" : "=A" (Var))
#elif defined __x86_64__
# define HP_TIMING_NOW(Var) \
  ({ unsigned int _hi, _lo; \
     asm volatile ("rdtsc" : "=a" (_lo), "=d" (_hi)); \
     (Var) = ((unsigned long long int) _hi << 32) | _lo; })
#elif defined __ia64__
#define HP_TIMING_NOW(Var)	__asm__ __volatile__ ("mov %0=ar.itc" : "=r" (Var) : : "memory")
#else
#error "HP_TIMING_NOW missing"
#endif

/* Get current value of CLOCK and store it in TP.  */
int
clock_gettime (clockid_t clock_id, struct timespec *tp)
{
  int retval = -1;

  switch (clock_id)
    {
    case CLOCK_PROCESS_CPUTIME_ID:
      {

	static hp_timing_t freq;
	hp_timing_t tsc;

	/* Get the current counter.  */
	HP_TIMING_NOW (tsc);

	if (freq == 0)
	  {
	    freq = get_clockfreq ();
	    if (freq == 0)
	      return EINVAL;
	  }

	/* Compute the seconds.  */
	tp->tv_sec = tsc / freq;

	/* And the nanoseconds.  This computation should be stable until
	   we get machines with about 16GHz frequency.  */
	tp->tv_nsec = ((tsc % freq) * UINT64_C (1000000000)) / freq;

	retval = 0;
      }
    break;

    default:
      errno = EINVAL;
      break;
    }

  return retval;
}
