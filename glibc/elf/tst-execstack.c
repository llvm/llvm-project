/* Test program for making nonexecutable stacks executable
   on load of a DSO that requires executable stacks.  */

#include <dlfcn.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <error.h>
#include <stackinfo.h>

static void
print_maps (void)
{
#if 0
  char *cmd = NULL;
  asprintf (&cmd, "cat /proc/%d/maps", getpid ());
  system (cmd);
  free (cmd);
#endif
}

static void deeper (void (*f) (void));

#if USE_PTHREADS
# include <pthread.h>

static void *
tryme_thread (void *f)
{
  (*((void (*) (void)) f)) ();

  return 0;
}

static pthread_barrier_t startup_barrier, go_barrier;
static void *
waiter_thread (void *arg)
{
  void **f = arg;
  pthread_barrier_wait (&startup_barrier);
  pthread_barrier_wait (&go_barrier);

  (*((void (*) (void)) *f)) ();

  return 0;
}
#endif

static bool allow_execstack = true;


static int
do_test (void)
{
  /* Check whether SELinux is enabled and disallows executable stacks.  */
  FILE *fp = fopen ("/selinux/enforce", "r");
  if (fp != NULL)
    {
      char *line = NULL;
      size_t linelen = 0;

      bool enabled = false;
      ssize_t n = getline (&line, &linelen, fp);
      if (n > 0 && line[0] != '0')
	enabled = true;

      fclose (fp);

      if (enabled)
	{
	  fp = fopen ("/selinux/booleans/allow_execstack", "r");
	  if (fp != NULL)
	    {
	      n = getline (&line, &linelen, fp);
	      if (n > 0 && line[0] == '0')
		allow_execstack = false;
	    }

	  fclose (fp);
	}
    }

  printf ("executable stacks %sallowed\n", allow_execstack ? "" : "not ");

  static void *f;		/* Address of this is used in other threads. */

#if USE_PTHREADS
  /* Create some threads while stacks are nonexecutable.  */
  #define N 5
  pthread_t thr[N];

  pthread_barrier_init (&startup_barrier, NULL, N + 1);
  pthread_barrier_init (&go_barrier, NULL, N + 1);

  for (int i = 0; i < N; ++i)
    {
      int rc = pthread_create (&thr[i], NULL, &waiter_thread, &f);
      if (rc)
	error (1, rc, "pthread_create");
    }

  /* Make sure they are all there using their stacks.  */
  pthread_barrier_wait (&startup_barrier);
  puts ("threads waiting");
#endif

  print_maps ();

#if USE_PTHREADS
  void *old_stack_addr, *new_stack_addr;
  size_t stack_size;
  pthread_t me = pthread_self ();
  pthread_attr_t attr;
  int ret = 0;

  ret = pthread_getattr_np (me, &attr);
  if (ret)
    {
      printf ("before execstack: pthread_getattr_np returned error: %s\n",
	      strerror (ret));
      return 1;
    }

  ret = pthread_attr_getstack (&attr, &old_stack_addr, &stack_size);
  if (ret)
    {
      printf ("before execstack: pthread_attr_getstack returned error: %s\n",
	      strerror (ret));
      return 1;
    }
# if _STACK_GROWS_DOWN
    old_stack_addr += stack_size;
# else
    old_stack_addr -= stack_size;
# endif
#endif

  /* Loading this module should force stacks to become executable.  */
  void *h = dlopen ("tst-execstack-mod.so", RTLD_LAZY);
  if (h == NULL)
    {
      printf ("cannot load: %s\n", dlerror ());
      return allow_execstack;
    }

  f = dlsym (h, "tryme");
  if (f == NULL)
    {
      printf ("symbol not found: %s\n", dlerror ());
      return 1;
    }

  /* Test if that really made our stack executable.
     The `tryme' function should crash if not.  */

  (*((void (*) (void)) f)) ();

  print_maps ();

#if USE_PTHREADS
  ret = pthread_getattr_np (me, &attr);
  if (ret)
    {
      printf ("after execstack: pthread_getattr_np returned error: %s\n",
	      strerror (ret));
      return 1;
    }

  ret = pthread_attr_getstack (&attr, &new_stack_addr, &stack_size);
  if (ret)
    {
      printf ("after execstack: pthread_attr_getstack returned error: %s\n",
	      strerror (ret));
      return 1;
    }

# if _STACK_GROWS_DOWN
    new_stack_addr += stack_size;
# else
    new_stack_addr -= stack_size;
# endif

  /* It is possible that the dlopen'd module may have been mmapped just below
     the stack.  The stack size is taken as MIN(stack rlimit size, end of last
     vma) in pthread_getattr_np.  If rlimit is set high enough, it is possible
     that the size may have changed.  A subsequent call to
     pthread_attr_getstack returns the size and (bottom - size) as the
     stacksize and stackaddr respectively.  If the size changes due to the
     above, then both stacksize and stackaddr can change, but the stack bottom
     should remain the same, which is computed as stackaddr + stacksize.  */
  if (old_stack_addr != new_stack_addr)
    {
      printf ("Stack end changed, old: %p, new: %p\n",
	      old_stack_addr, new_stack_addr);
      return 1;
    }
  printf ("Stack address remains the same: %p\n", old_stack_addr);
#endif

  /* Test that growing the stack region gets new executable pages too.  */
  deeper ((void (*) (void)) f);

  print_maps ();

#if USE_PTHREADS
  /* Test that a fresh thread now gets an executable stack.  */
  {
    pthread_t th;
    int rc = pthread_create (&th, NULL, &tryme_thread, f);
    if (rc)
      error (1, rc, "pthread_create");
  }

  puts ("threads go");
  /* The existing threads' stacks should have been changed.
     Let them run to test it.  */
  pthread_barrier_wait (&go_barrier);

  pthread_exit ((void *) (long int) (! allow_execstack));
#endif

  return ! allow_execstack;
}

static void
deeper (void (*f) (void))
{
  char stack[1100 * 1024];
  memfrob (stack, sizeof stack);
  (*f) ();
  memfrob (stack, sizeof stack);
}


#include <support/test-driver.c>
