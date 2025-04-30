/* Test case by Hui Huang <hui.huang@sun.com>.  */
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>


static void *
start_routine (void *args)
{
  int i;
  void **addrs = (void **) args;
  for (i = 0; i < 10000; ++i)
    addrs[i % 1024] = dlsym (NULL, "does_not_exist");

  return addrs;
}


static int
do_test (void)
{
  pthread_t tid1, tid2, tid3;

  void *addrs1[1024];
  void *addrs2[1024];
  void *addrs3[1024];

  if (pthread_create (&tid1, NULL, start_routine, addrs1) != 0)
    {
      puts ("1st create failed");
      exit (1);
    }
  if (pthread_create (&tid2, NULL, start_routine, addrs2) != 0)
    {
      puts ("2nd create failed");
      exit (1);
    }
  if (pthread_create (&tid3, NULL, start_routine, addrs3) != 0)
    {
      puts ("3rd create failed");
      exit (1);
    }

  if (pthread_join (tid1, NULL) != 0)
    {
      puts ("1st join failed");
      exit (1);
    }
  if (pthread_join (tid2, NULL) != 0)
    {
      puts ("2nd join failed");
      exit (1);
    }
  if (pthread_join (tid3, NULL) != 0)
    {
      puts ("2rd join failed");
      exit (1);
    }

  return 0;
}


#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
