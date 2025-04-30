#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <stdint.h>

volatile int gotit = 0;

static void
alarm_handler (int signal)
{
    gotit = 1;
}


int
main (int argc, char ** argv)
{
  clock_t start, stop;

  if (signal(SIGALRM, alarm_handler) == SIG_ERR)
    {
      perror ("signal");
      exit (1);
    }
  alarm(1);
  start = clock ();
  while (!gotit);
  stop = clock ();

  printf ("%jd clock ticks per second (start=%jd,stop=%jd)\n",
	  (intmax_t) (stop - start), (intmax_t) start, (intmax_t) stop);
  printf ("CLOCKS_PER_SEC=%jd, sysconf(_SC_CLK_TCK)=%ld\n",
	  (intmax_t) CLOCKS_PER_SEC, sysconf(_SC_CLK_TCK));
  return 0;
}
