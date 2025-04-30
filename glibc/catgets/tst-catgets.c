#include <assert.h>
#include <mcheck.h>
#include <nl_types.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>


static const char *msgs[] =
{
#define INPUT(str)
#define OUTPUT(str) str,
#include <intl/msgs.h>
};
#define nmsgs (sizeof (msgs) / sizeof (msgs[0]))


/* Test for unbounded alloca.  */
static int
do_bz17905 (void)
{
  char *buf;
  struct rlimit rl;
  nl_catd result __attribute__ ((unused));

  const int sz = 1024 * 1024;

  getrlimit (RLIMIT_STACK, &rl);
  rl.rlim_cur = sz;
  setrlimit (RLIMIT_STACK, &rl);

  buf = malloc (sz + 1);
  memset (buf, 'A', sz);
  buf[sz] = '\0';
  setenv ("NLSPATH", buf, 1);

  result = catopen (buf, NL_CAT_LOCALE);
  assert (result == (nl_catd) -1);

  free (buf);
  return 0;
}

#define ROUNDS 5

static int
do_test (void)
{
  int rnd;
  int result = 0;

  mtrace ();

  /* We do this a few times to stress the memory handling.  */
  for (rnd = 0; rnd < ROUNDS; ++rnd)
    {
      nl_catd cd = catopen ("libc", 0);
      size_t cnt;

      if (cd == (nl_catd) -1)
	{
	  printf ("cannot load catalog: %m\n");
	  result = 1;
	  break;
	}

      /* Go through all the messages and compare the result.  */
      for (cnt = 0; cnt < nmsgs; ++cnt)
	{
	  char *trans;

	  trans = catgets (cd, 1, 1 + cnt,
			   "+#+# if this comes backs it's an error");

	  if (trans == NULL)
	    {
	      printf ("catgets return NULL for %zd\n", cnt);
	      result = 1;
	    }
	  else if (strcmp (trans, msgs[cnt]) != 0 && msgs[cnt][0] != '\0')
	    {
	      printf ("expected \"%s\", got \"%s\"\n", msgs[cnt], trans);
	      result = 1;
	    }
	}

      if (catclose (cd) != 0)
	{
	  printf ("catclose failed: %m\n");
	  result = 1;
	}
    }

  result += do_bz17905 ();
  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
