#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/resource.h>

static void use_stack (size_t needed);

void (*use_stack_ptr) (size_t) = use_stack;

static void
use_stack (size_t needed)
{
  size_t sz = sysconf (_SC_PAGESIZE);
  char *buf = alloca (sz);
  memset (buf, '\0', sz);

  if (needed > sz)
    use_stack_ptr (needed  - sz);
}

static void
use_up_memory (void)
{
  struct rlimit rl;
  getrlimit (RLIMIT_AS, &rl);
  rl.rlim_cur = 10 * 1024 * 1024;
  setrlimit (RLIMIT_AS, &rl);

  char *c;
  int PAGESIZE = getpagesize ();
  while (1)
    {
      c = mmap (NULL, PAGESIZE, PROT_NONE, MAP_ANON | MAP_PRIVATE, -1, 0);
      if (c == MAP_FAILED)
	break;
    }
}

static void *
child (void *arg)
{
  sleep (1);
  return arg;
}

static int
do_test (void)
{
  int err;
  pthread_t tid;

  /* Allocate the memory needed for the stack.  */
#ifdef PTHREAD_STACK_MIN
  use_stack_ptr (PTHREAD_STACK_MIN);
#else
  use_stack_ptr (4 * getpagesize ());
#endif

  use_up_memory ();

  err = pthread_create (&tid, NULL, child, NULL);
  if (err != 0)
    {
      printf ("pthread_create returns %d: %s\n", err,
	      err == EAGAIN ? "OK" : "FAIL");
      return err != EAGAIN;
    }

  /* We did not fail to allocate memory despite the preparation.  Oh well.  */
  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
