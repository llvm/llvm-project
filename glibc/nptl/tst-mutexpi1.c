#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>


static pthread_mutexattr_t a;

static void
prepare (void)
{
  if (pthread_mutexattr_init (&a) != 0)
    {
      puts ("mutexattr_init failed");
      exit (1);
    }

  if (pthread_mutexattr_setprotocol (&a, PTHREAD_PRIO_INHERIT) != 0)
    {
      puts ("mutexattr_setprotocol failed");
      exit (1);
    }
}
#define PREPARE(argc, argv) prepare ()


#define ATTR &a
#define ATTR_NULL false
#include "tst-mutex1.c"
