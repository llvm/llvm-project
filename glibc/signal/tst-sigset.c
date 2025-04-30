/* Test sig*set functions.  */

#include <signal.h>

#include <support/check.h>

static int
do_test (void)
{
  sigset_t set;
  TEST_VERIFY (sigemptyset (&set) == 0);

#define VERIFY(set, sig)			\
  TEST_VERIFY (sigismember (&set, sig) == 0);	\
  TEST_VERIFY (sigaddset (&set, sig) == 0);	\
  TEST_VERIFY (sigismember (&set, sig) != 0);	\
  TEST_VERIFY (sigdelset (&set, sig) == 0);	\
  TEST_VERIFY (sigismember (&set, sig) == 0)

  /* ISO C99 signals.  */
  VERIFY (set, SIGINT);
  VERIFY (set, SIGILL);
  VERIFY (set, SIGABRT);
  VERIFY (set, SIGFPE);
  VERIFY (set, SIGSEGV);
  VERIFY (set, SIGTERM);

  /* Historical signals specified by POSIX. */
  VERIFY (set, SIGHUP);
  VERIFY (set, SIGQUIT);
  VERIFY (set, SIGTRAP);
  VERIFY (set, SIGKILL);
  VERIFY (set, SIGBUS);
  VERIFY (set, SIGSYS);
  VERIFY (set, SIGPIPE);
  VERIFY (set, SIGALRM);

  /* New(er) POSIX signals (1003.1-2008, 1003.1-2013).  */
  VERIFY (set, SIGURG);
  VERIFY (set, SIGSTOP);
  VERIFY (set, SIGTSTP);
  VERIFY (set, SIGCONT);
  VERIFY (set, SIGCHLD);
  VERIFY (set, SIGTTIN);
  VERIFY (set, SIGTTOU);
  VERIFY (set, SIGPOLL);
  VERIFY (set, SIGXCPU);
  VERIFY (set, SIGXFSZ);
  VERIFY (set, SIGVTALRM);
  VERIFY (set, SIGPROF);
  VERIFY (set, SIGUSR1);
  VERIFY (set, SIGUSR2);

  /* Nonstandard signals found in all modern POSIX systems
     (including both BSD and Linux).  */
  VERIFY (set, SIGWINCH);

  /* Arch-specific signals.  */
#ifdef SIGEMT
  VERIFY (set, SIGEMT);
#endif
#ifdef SIGLOST
  VERIFY (set, SIGLOST);
#endif
#ifdef SIGINFO
  VERIFY (set, SIGINFO);
#endif
#ifdef SIGSTKFLT
  VERIFY (set, SIGSTKFLT);
#endif
#ifdef SIGPWR
  VERIFY (set, SIGPWR);
#endif

  /* Read-time signals (POSIX.1b real-time extensions).  If they are
     supported SIGRTMAX value is greater than SIGRTMIN.  */
  for (int rtsig = SIGRTMIN; rtsig <= SIGRTMAX; rtsig++)
    {
      VERIFY (set, rtsig);
    }

  return 0;
}

#include <support/test-driver.c>
