#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int win = 0;

static void
handler (int sig)
{
  printf ("Received signal %d (%s).\n", sig, strsignal(sig));
  win = 1;
}

int
main (void)
{
  if (signal (SIGTERM, handler) == SIG_ERR)
    {
      perror ("signal: SIGTERM");
      exit (EXIT_FAILURE);
    }

  puts ("Set handler.");

  printf ("Sending myself signal %d.\n", SIGTERM);
  fflush (stdout);

  if (raise (SIGTERM) < 0)
    {
      perror ("raise: SIGTERM");
      exit (EXIT_FAILURE);
    }

  if (!win)
    {
      puts ("Didn't get any signal.  Test FAILED!");
      exit (EXIT_FAILURE);
    }

  puts ("Got a signal.  Test succeeded.");

  return EXIT_SUCCESS;
}
