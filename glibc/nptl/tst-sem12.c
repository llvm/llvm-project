#include <time.h>
#include <sys/time.h>


#define PREPARE \
  struct timespec ts; \
  struct timeval tv; \
  gettimeofday (&tv, NULL); \
  TIMEVAL_TO_TIMESPEC (&tv, &ts); \
  ts.tv_sec += 60;

#define SEM_WAIT(s) sem_timedwait (s, &ts)

#include "tst-sem11.c"
