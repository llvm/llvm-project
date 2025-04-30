/*
 * This fcrypt/crypt speed testing program
 * is derived from one floating around in
 * the net. It's distributed along with
 * UFC-crypt but is not covered by any
 * licence.
 *
 * @(#)speeds.c	1.11 20 Aug 1996
 */

#include <signal.h>
#include <stdio.h>

#ifndef SIGVTALRM
/*
 * patch from chip@chinacat.unicom.com (Chip Rosenthal):
 * you may enable it if your system does not include
 * a setitimer() function. You'll have to ensure the
 * existence an environment variable: HZ giving how many
 * ticks goes per second.
 * If not existing in your default environment 50, 60
 * or even 100 may be the right value. Perhaps you should
 * then use 'time ./ufc 10000' instead of guessing.
 */
#define NO_ITIMER
#endif

#ifdef NO_ITIMER
#include <sys/types.h>
#include <sys/times.h>
#else
#include <sys/time.h>
#endif

static int cnt;
#ifdef NO_ITIMER
char *hz;
struct tms tstart, tfinish;
#endif
#define ITIME	10		/* Number of seconds to run test. */

char *crypt(), *fcrypt();

void
Stop (void)
{
    double elapsed;
#ifdef NO_ITIMER
    (void) times(&tfinish);
    elapsed = ((tfinish.tms_utime + tfinish.tms_stime) -
	(tstart.tms_utime + tstart.tms_stime)) / atoi(hz);
    printf("elapsed time = %d sec,  CPU time = %f sec\n", ITIME, elapsed);
#else
    elapsed = ITIME;
#endif
    printf ("Did %f %s()s per second.\n", ((float) cnt) / elapsed,
#if defined(FCRYPT)
	    "fcrypt"
#else
	    "crypt"
#endif
    );
    exit (0);
}

/*
 * Silly rewrite of 'bzero'. I do so
 * because some machines don't have
 * bzero and some don't have memset.
 */

static void clearmem(start, cnt)
  char *start;
  int cnt;
  { while(cnt--)
      *start++ = '\0';
  }

main (void)
{
   char *s;
#ifdef NO_ITIMER
    extern char *getenv();
#else
    struct itimerval itv;
#endif

#ifdef NO_ITIMER
    if ((hz = getenv("HZ")) == NULL) {
	fprintf(stderr, "HZ environment parameter undefined\n");
	exit(1);
    }
#endif

#ifdef FCRYPT
    printf("\n");
    printf("Warning: this version of the speed program may run slower when\n");
    printf("benchmarking UFC-crypt than previous versions. This is because it\n");
    printf("stresses the CPU hardware cache in order to get benchmark figures\n");
    printf("that corresponds closer to the performance that can be expected in\n");
    printf("a password cracker.\n\n");
#endif

    printf ("Running %s for %d seconds of virtual time ...\n",
#ifdef FCRYPT
    "UFC-crypt",
#else
    "crypt(libc)",
#endif
	    ITIME);

#ifdef FCRYPT
    init_des ();
#endif

#ifdef NO_ITIMER
    signal(SIGALRM, Stop);
    switch (fork()) {
    case -1:
	perror("fork failed");
	exit(1);
    case 0:
	sleep(10);
	kill(getppid(), SIGALRM);
	exit(0);
    default:
	(void) times(&tstart);
    }
#else
    clearmem ((char*)&itv, (int)sizeof (itv));
    signal (SIGVTALRM, Stop);
    itv.it_value.tv_sec = ITIME;
    itv.it_value.tv_usec = 0;
    setitimer (ITIMER_VIRTUAL, &itv, NULL);
#endif


    s = "fredred";
    for (cnt = 0;; cnt++)
    {
#ifdef FCRYPT
	s = fcrypt (s, "eek");
#else
	s = crypt (s, "eek");
#endif
    }
}






