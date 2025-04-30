#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>


static pid_t pid;

static void *
pid_thread (void *arg)
{
  if (pid != getpid ())
    {
      printf ("pid wrong in thread: should be %d, is %d\n",
	      (int) pid, (int) getpid ());
      return (void *) 1L;
    }

  return NULL;
}

static int
do_test (void)
{
  pid = getpid ();

  pthread_t thr;
  int ret = pthread_create (&thr, NULL, pid_thread, NULL);
  if (ret)
    {
      printf ("pthread_create failed: %d\n", ret);
      return 1;
    }

  void *thr_ret;
  ret = pthread_join (thr, &thr_ret);
  if (ret)
    {
      printf ("pthread_create failed: %d\n", ret);
      return 1;
    }
  else if (thr_ret)
    {
      printf ("thread getpid failed\n");
      return 1;
    }

  pid_t child = fork ();
  if (child == -1)
    {
      printf ("fork failed: %m\n");
      return 1;
    }
  else if (child == 0)
    {
      if (pid == getpid ())
	{
	  puts ("pid did not change after fork");
	  exit (1);
	}

      pid = getpid ();
      ret = pthread_create (&thr, NULL, pid_thread, NULL);
      if (ret)
	{
	  printf ("pthread_create failed: %d\n", ret);
	  return 1;
	}

      ret = pthread_join (thr, &thr_ret);
      if (ret)
	{
	  printf ("pthread_create failed: %d\n", ret);
	  return 1;
	}
      else if (thr_ret)
	{
	  printf ("thread getpid failed\n");
	  return 1;
	}

      return 0;
    }

  int status;
  if (TEMP_FAILURE_RETRY (waitpid (child, &status, 0)) != child)
    {
      puts ("waitpid failed");
      kill (child, SIGKILL);
      return 1;
    }

  if (!WIFEXITED (status))
    {
      if (WIFSIGNALED (status))
	printf ("died from signal %s\n", strsignal (WTERMSIG (status)));
      else
	puts ("did not terminate correctly");
      return 1;
    }
  if (WEXITSTATUS (status) != 0)
    {
      printf ("exit code %d\n", WEXITSTATUS (status));
      return 1;
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
