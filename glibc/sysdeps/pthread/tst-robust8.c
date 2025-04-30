#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>

#include <pthreadP.h>



static void prepare (void);
#define PREPARE(argc, argv) prepare ()
static int do_test (void);
#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"


static int fd;
#define N 100

static void
prepare (void)
{
  fd = create_temp_file ("tst-robust8", NULL);
  if (fd == -1)
    exit (1);
}


#define THESIGNAL SIGKILL
#define ROUNDS 5
#define THREADS 9


static const struct timespec before = { 0, 0 };


static pthread_mutex_t *map;


static void *
tf (void *arg)
{
  long int nr = (long int) arg;
  int fct = nr % 3;

  uint8_t state[N];
  memset (state, '\0', sizeof (state));

  while (1)
    {
      int r = random () % N;
      if (state[r] == 0)
	{
	  int e;

	  switch (fct)
	    {
	    case 0:
	      e = pthread_mutex_lock (&map[r]);
	      if (e != 0)
		{
		  printf ("mutex_lock of %d in thread %ld failed with %d\n",
			  r, nr, e);
		  exit (1);
		}
	      state[r] = 1;
	      break;
	    case 1:
	      e = pthread_mutex_timedlock (&map[r], &before);
	      if (e != 0 && e != ETIMEDOUT)
		{
		  printf ("\
mutex_timedlock of %d in thread %ld failed with %d\n",
			  r, nr, e);
		  exit (1);
		}
	      break;
	    default:
	      e = pthread_mutex_trylock (&map[r]);
	      if (e != 0 && e != EBUSY)
		{
		  printf ("mutex_trylock of %d in thread %ld failed with %d\n",
			  r, nr, e);
		  exit (1);
		}
	      break;
	    }

	  if (e == EOWNERDEAD)
	    pthread_mutex_consistent (&map[r]);

	  if (e == 0 || e == EOWNERDEAD)
	    state[r] = 1;
	}
      else
	{
	  int e = pthread_mutex_unlock (&map[r]);
	  if (e != 0)
	    {
	      printf ("mutex_unlock of %d in thread %ld failed with %d\n",
		      r, nr, e);
	      exit (1);
	    }

	  state[r] = 0;
	}
    }
}


static void
child (int round)
{
  for (int thread = 1; thread <= THREADS; ++thread)
    {
      pthread_t th;
      if (pthread_create (&th, NULL, tf, (void *) (long int) thread) != 0)
	{
	  printf ("cannot create thread %d in round %d\n", thread, round);
	  exit (1);
	}
    }

  struct timespec ts;
  ts.tv_sec = 0;
  ts.tv_nsec = 1000000000 / ROUNDS;
  while (nanosleep (&ts, &ts) != 0)
    /* nothing */;

  /* Time to die.  */
  kill (getpid (), THESIGNAL);

  /* We better never get here.  */
  abort ();
}


static int
do_test (void)
{
  if (ftruncate (fd, N * sizeof (pthread_mutex_t)) != 0)
    {
      puts ("cannot size new file");
      return 1;
    }

  map = mmap (NULL, N * sizeof (pthread_mutex_t), PROT_READ | PROT_WRITE,
	      MAP_SHARED, fd, 0);
  if (map == MAP_FAILED)
    {
      puts ("mapping failed");
      return 1;
    }

  pthread_mutexattr_t ma;
  if (pthread_mutexattr_init (&ma) != 0)
    {
      puts ("mutexattr_init failed");
      return 0;
    }
  if (pthread_mutexattr_setrobust (&ma, PTHREAD_MUTEX_ROBUST_NP) != 0)
    {
      puts ("mutexattr_setrobust failed");
      return 1;
    }
  if (pthread_mutexattr_setpshared (&ma, PTHREAD_PROCESS_SHARED) != 0)
    {
      puts ("mutexattr_setpshared failed");
      return 1;
    }
#ifdef ENABLE_PI
  if (pthread_mutexattr_setprotocol (&ma, PTHREAD_PRIO_INHERIT) != 0)
    {
      puts ("pthread_mutexattr_setprotocol failed");
      return 1;
    }
#endif

  for (int round = 1; round <= ROUNDS; ++round)
    {
      for (int n = 0; n < N; ++n)
	{
	  int e = pthread_mutex_init (&map[n], &ma);
	  if (e == ENOTSUP)
	    {
#ifdef ENABLE_PI
	      puts ("cannot support pshared robust PI mutexes");
#else
	      puts ("cannot support pshared robust mutexes");
#endif
	      return 0;
	    }
	  if (e != 0)
	    {
	      printf ("mutex_init %d in round %d failed\n", n + 1, round);
	      return 1;
	    }
	}

      pid_t p = fork ();
      if (p == -1)
	{
	  printf ("fork in round %d failed\n", round);
	  return 1;
	}
      if (p == 0)
	child (round);

      int status;
      if (TEMP_FAILURE_RETRY (waitpid (p, &status, 0)) != p)
	{
	  printf ("waitpid in round %d failed\n", round);
	  return 1;
	}
      if (!WIFSIGNALED (status))
	{
	  printf ("child did not die of a signal in round %d\n", round);
	  return 1;
	}
      if (WTERMSIG (status) != THESIGNAL)
	{
	  printf ("child did not die of signal %d in round %d\n",
		  THESIGNAL, round);
	  return 1;
	}

      for (int n = 0; n < N; ++n)
	{
	  int e = pthread_mutex_lock (&map[n]);
	  if (e != 0 && e != EOWNERDEAD)
	    {
	      printf ("mutex_lock %d failed in round %d\n", n + 1, round);
	      return 1;
	    }
	}

      for (int n = 0; n < N; ++n)
	if (pthread_mutex_unlock (&map[n]) != 0)
	  {
	    printf ("mutex_unlock %d failed in round %d\n", n + 1, round);
	    return 1;
	  }

      for (int n = 0; n < N; ++n)
	{
	  int e = pthread_mutex_destroy (&map[n]);
	  if (e != 0)
	    {
	      printf ("mutex_destroy %d in round %d failed with %d\n",
		      n + 1, round, e);
#ifdef __PTHREAD_NPTL
	      printf("nusers = %d\n", (int) map[n].__data.__nusers);
#endif
	      return 1;
	    }
	}
    }

  if (pthread_mutexattr_destroy (&ma) != 0)
    {
      puts ("mutexattr_destroy failed");
      return 1;
    }

  if (munmap (map, N * sizeof (pthread_mutex_t)) != 0)
    {
      puts ("munmap failed");
      return 1;
    }

  return 0;
}
