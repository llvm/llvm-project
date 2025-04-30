#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#include "tst-tpp.h"

static pthread_mutexattr_t a;

static void
prepare (void)
{
  init_tpp_test ();

  if (pthread_mutexattr_init (&a) != 0)
    {
      puts ("mutexattr_init failed");
      exit (1);
    }

  if (pthread_mutexattr_setprotocol (&a, PTHREAD_PRIO_PROTECT) != 0)
    {
      puts ("mutexattr_setprotocol failed");
      exit (1);
    }

  if (pthread_mutexattr_setprioceiling (&a, 6) != 0)
    {
      puts ("mutexattr_setprioceiling failed");
      exit (1);
    }
}
#define PREPARE(argc, argv) prepare ()

static int do_test (void);

static int
do_test_wrapper (void)
{
  init_tpp_test ();
  return do_test ();
}
#define TEST_FUNCTION do_test_wrapper ()

#define ATTR &a
#define ATTR_NULL false
#include "tst-mutex6.c"
