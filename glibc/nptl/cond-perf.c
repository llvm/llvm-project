#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <atomic.h>

static pthread_cond_t cond1 = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t mut1 = PTHREAD_MUTEX_INITIALIZER;

static pthread_cond_t cond2 = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t mut2 = PTHREAD_MUTEX_INITIALIZER;

static bool last_round;
static int ntogo;
static bool alldone;


static void *
cons (void *arg)
{
  pthread_mutex_lock (&mut1);

  do
    {
      if (atomic_decrement_and_test (&ntogo))
	{
	  pthread_mutex_lock (&mut2);
	  alldone = true;
	  pthread_cond_signal (&cond2);
	  pthread_mutex_unlock (&mut2);
	}

      pthread_cond_wait (&cond1, &mut1);
    }
  while (! last_round);

  pthread_mutex_unlock (&mut1);

  return NULL;
}


int
main (int argc, char *argv[])
{
  int opt;
  int err;
  int nthreads = 10;
  int nrounds = 100;
  bool keeplock = false;

  while ((opt = getopt (argc, argv, "n:r:k")) != -1)
    switch (opt)
      {
      case 'n':
	nthreads = atol (optarg);
	break;
      case 'r':
	nrounds = atol (optarg);
	break;
      case 'k':
	keeplock = true;
	break;
      }

  ntogo = nthreads;

  pthread_t th[nthreads];
  int i;
  for (i = 0; __builtin_expect (i < nthreads, 1); ++i)
    if (__glibc_unlikely ((err = pthread_create (&th[i], NULL, cons, (void *) (long) i)) != 0))
      printf ("pthread_create: %s\n", strerror (err));

  for (i = 0; __builtin_expect (i < nrounds, 1); ++i)
    {
      pthread_mutex_lock (&mut2);
      while (! alldone)
	pthread_cond_wait (&cond2, &mut2);
      pthread_mutex_unlock (&mut2);

      pthread_mutex_lock (&mut1);
      if (! keeplock)
	pthread_mutex_unlock (&mut1);

      ntogo = nthreads;
      alldone = false;
      if (i + 1 >= nrounds)
	last_round = true;

      pthread_cond_broadcast (&cond1);

      if (keeplock)
	pthread_mutex_unlock (&mut1);
    }

  for (i = 0; i < nthreads; ++i)
    if ((err = pthread_join (th[i], NULL)) != 0)
      printf ("pthread_create: %s\n", strerror (err));

  return 0;
}
