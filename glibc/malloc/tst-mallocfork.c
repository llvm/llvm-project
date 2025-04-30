/* Derived from the test case in
   https://sourceware.org/bugzilla/show_bug.cgi?id=838.  */
#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <libc-diag.h>

static void
sig_handler (int signum)
{
  pid_t child = fork ();
  if (child == 0)
    exit (0);
  TEMP_FAILURE_RETRY (waitpid (child, NULL, 0));
}

static int
do_test (void)
{
  pid_t parent = getpid ();

  struct sigaction action = { .sa_handler = sig_handler };
  sigemptyset (&action.sa_mask);

  DIAG_PUSH_NEEDS_COMMENT;
  DIAG_IGNORE_NEEDS_COMMENT (10, "-Wunused-result");
  /* The result of malloc is deliberately ignored, so do not warn
     about that.  */
  malloc (sizeof (int));
  DIAG_POP_NEEDS_COMMENT;

  if (sigaction (SIGALRM, &action, NULL) != 0)
    {
      puts ("sigaction failed");
      return 1;
    }

  /* Create a child that sends the signal to be caught.  */
  pid_t child = fork ();
  if (child == 0)
    {
      if (kill (parent, SIGALRM) == -1)
	perror ("kill");
      exit (0);
    }

  TEMP_FAILURE_RETRY (waitpid (child, NULL, 0));

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
