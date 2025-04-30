#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>


pthread_mutexattr_t a;
pthread_mutexattr_t *attr;

static void
prepare (void)
{
  attr = &a;
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


#define ATTR attr
#include "tst-mutex6.c"
