/* Tests for POSIX timer implementation using another process's CPU clock.  */

#include <unistd.h>

#if _POSIX_THREADS && defined _POSIX_CPUTIME

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <time.h>
#include <signal.h>
#include <sys/wait.h>

static clockid_t child_clock;

#define TEST_CLOCK child_clock
#define TEST_CLOCK_MISSING(clock) \
  (setup_test () ? "other-process CPU clock timer support" : NULL)

/* This function is intended to rack up both user and system time.  */
static void
chew_cpu (void)
{
  while (1)
    {
      static volatile char buf[4096];
      for (int i = 0; i < 100; ++i)
	for (size_t j = 0; j < sizeof buf; ++j)
	  buf[j] = 0xaa;
      int nullfd = open ("/dev/null", O_WRONLY);
      for (int i = 0; i < 100; ++i)
	for (size_t j = 0; j < sizeof buf; ++j)
	  buf[j] = 0xbb;
      write (nullfd, (char *) buf, sizeof buf);
      close (nullfd);
      if (getppid () == 1)
	_exit (2);
    }
}

static pid_t child;
static void
cleanup_child (void)
{
  if (child <= 0)
    return;
  if (kill (child, SIGKILL) < 0 && errno != ESRCH)
    printf ("cannot kill child %d: %m\n", child);
  else
    {
      int status;
      errno = 0;
      if (waitpid (child, &status, 0) != child)
	printf ("waitpid %d: %m\n", child);
    }
}
#define CLEANUP_HANDLER cleanup_child ()

static int
setup_test (void)
{
  /* Test timers on a process CPU clock by having a child process eating
     CPU.  First make sure we can make such timers at all.  */

  int pipefd[2];
  if (pipe (pipefd) < 0)
    {
      printf ("pipe: %m\n");
      exit (1);
    }

  child = fork ();

  if (child == 0)
    {
      char c;
      close (pipefd[1]);
      if (read (pipefd[0], &c, 1) == 1)
	chew_cpu ();
      _exit (1);
    }

  if (child < 0)
    {
      printf ("fork: %m\n");
      exit (1);
    }

  atexit (&cleanup_child);

  close (pipefd[0]);

  int e = clock_getcpuclockid (child, &child_clock);
  if (e == EPERM)
    {
      puts ("clock_getcpuclockid does not support other processes");
      return 1;
    }
  if (e != 0)
    {
      printf ("clock_getcpuclockid: %s\n", strerror (e));
      exit (1);
    }

  timer_t t;
  if (timer_create (TEST_CLOCK, NULL, &t) != 0)
    {
      printf ("timer_create: %m\n");
      return 1;
    }
  timer_delete (t);

  /* Get the child started chewing.  */
  if (write (pipefd[1], "x", 1) != 1)
    {
      printf ("write to pipe: %m\n");
      return 1;
    }
  close (pipefd[1]);

  return 0;
}

#else
# define TEST_CLOCK_MISSING(clock) "process clocks"
#endif

#include "tst-timer4.c"
