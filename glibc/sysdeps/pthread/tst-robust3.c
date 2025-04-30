#include <time.h>
#include <sys/time.h>


static struct timespec tmo;


#define PREPARE_TMO \
  do {									      \
    struct timeval tv;							      \
    gettimeofday (&tv, NULL);						      \
									      \
    /* Define the timeout as one hour in the future.  */		      \
    tmo.tv_sec = tv.tv_sec + 3600;					      \
    tmo.tv_nsec = 0;							      \
  } while (0)


#define LOCK(m) pthread_mutex_timedlock (m, &tmo)
#include "tst-robust1.c"
