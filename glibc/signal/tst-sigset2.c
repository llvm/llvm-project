/* sigset_SIG_HOLD_bug.c [BZ #1951] */
#include <errno.h>
#include <error.h>
#include <inttypes.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <libc-diag.h>

/* The sigset function is deprecated.  */
DIAG_IGNORE_NEEDS_COMMENT (4.9, "-Wdeprecated-declarations");

#define TEST_SIG SIGINT


/* Print mask of blocked signals for this process */
static void
printSigMask (const char *msg)
{
  sigset_t currMask;
  int sig;
  int cnt;

  if (msg != NULL)
    printf ("%s", msg);

  if (sigprocmask (SIG_BLOCK, NULL, &currMask) == -1)
    error (1, errno, "sigaction");

  cnt = 0;
  for (sig = 1; sig < NSIG; sig++)
    {
      if (sigismember (&currMask, sig))
	{
	  cnt++;
	  printf ("\t\t%d (%s)\n", sig, strsignal (sig));
        }
    }

  if (cnt == 0)
    printf ("\t\t<empty signal set>\n");
} /* printSigMask */

static void
handler (int sig)
{
  printf ("Caught signal %d\n", sig);
  printSigMask ("Signal mask in handler\n");
  printf ("Handler returning\n");
  _exit (1);
} /* handler */

static void
printDisposition (sighandler_t disp)
{
  if (disp == SIG_HOLD)
    printf ("SIG_HOLD");
  else if (disp == SIG_DFL)
    printf ("SIG_DFL");
  else if (disp == SIG_IGN)
    printf ("SIG_IGN");
  else
    printf ("handled at %" PRIxPTR, (uintptr_t) disp);
} /* printDisposition */

static int
returnTest1 (void)
{
  sighandler_t prev;

  printf ("===== TEST 1 =====\n");
  printf ("Blocking signal with sighold()\n");
  if (sighold (TEST_SIG) == -1)
    error (1, errno, "sighold");
  printSigMask ("Signal mask after sighold()\n");

  printf ("About to use sigset() to establish handler\n");
  prev = sigset (TEST_SIG, handler);
  if (prev == SIG_ERR)
    error(1, errno, "sigset");

  printf ("Previous disposition: ");
  printDisposition (prev);
  printf (" (should be SIG_HOLD)\n");
  if (prev != SIG_HOLD)
    {
      printf("TEST FAILED!!!\n");
      return 1;
    }
  return 0;
} /* returnTest1 */

static int
returnTest2 (void)
{
  sighandler_t prev;

  printf ("\n===== TEST 2 =====\n");

  printf ("About to use sigset() to set SIG_HOLD\n");
  prev = sigset (TEST_SIG, SIG_HOLD);
  if (prev == SIG_ERR)
    error (1, errno, "sigset");

  printf ("Previous disposition: ");
  printDisposition (prev);
  printf (" (should be SIG_DFL)\n");
  if (prev != SIG_DFL)
    {
      printf("TEST FAILED!!!\n");
      return 1;
    }
  return 0;
} /* returnTest2 */

static int
returnTest3 (void)
{
  sighandler_t prev;

  printf ("\n===== TEST 3 =====\n");

  printf ("About to use sigset() to set SIG_HOLD\n");
  prev = sigset (TEST_SIG, SIG_HOLD);
  if (prev == SIG_ERR)
    error (1, errno, "sigset");

  printf ("About to use sigset() to set SIG_HOLD (again)\n");
  prev = sigset (TEST_SIG, SIG_HOLD);
  if (prev == SIG_ERR)
    error (1, errno, "sigset");

  printf ("Previous disposition: ");
  printDisposition (prev);
  printf (" (should be SIG_HOLD)\n");
  if (prev != SIG_HOLD)
    {
      printf("TEST FAILED!!!\n");
      return 1;
    }
  return 0;
} /* returnTest3 */

int
main (int argc, char *argv[])
{
  pid_t childPid;

  childPid = fork();
  if (childPid == -1)
    error (1, errno, "fork");

  if (childPid == 0)
    exit (returnTest1 ());

  int status;
  if (TEMP_FAILURE_RETRY (waitpid (childPid, &status, 0)) != childPid)
    error (1, errno, "waitpid");
  int result = !WIFEXITED (status) || WEXITSTATUS (status) != 0;

  childPid = fork();
  if (childPid == -1)
    error (1, errno, "fork");

  if (childPid == 0)
    exit (returnTest2 ());

  if (TEMP_FAILURE_RETRY (waitpid (childPid, &status, 0)) != childPid)
    error (1, errno, "waitpid");
  result |= !WIFEXITED (status) || WEXITSTATUS (status) != 0;

  childPid = fork();
  if (childPid == -1)
    error (1, errno, "fork");

  if (childPid == 0)
    exit (returnTest3 ());

  if (TEMP_FAILURE_RETRY (waitpid (childPid, &status, 0)) != childPid)
    error (1, errno, "waitpid");
  result |= !WIFEXITED (status) || WEXITSTATUS (status) != 0;

  return result;
} /* main */
