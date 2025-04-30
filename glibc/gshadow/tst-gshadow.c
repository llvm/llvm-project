#include <gshadow.h>
#include <stdio.h>
#include <string.h>


static const struct sgrp data[] =
  {
    { (char *) "one", (char *) "pwdone",
      (char *[]) { (char *) "admoneone", (char *) "admonetwo",
		  (char *) "admonethree", NULL },
      (char *[]) { (char *) "memoneone", (char *) "memonetwo",
		  (char *) "memonethree", NULL } },
    { (char *) "two", (char *) "pwdtwo",
      (char *[]) { (char *) "admtwoone", (char *) "admtwotwo", NULL },
      (char *[]) { (char *) "memtwoone", (char *) "memtwotwo",
		  (char *) "memtwothree", NULL } },
    { (char *) "three", (char *) "pwdthree",
      (char *[]) { (char *) "admthreeone", (char *) "admthreetwo", NULL },
      (char *[]) { (char *) "memthreeone", (char *) "memthreetwo", NULL } },
    { (char *) "four", (char *) "pwdfour",
      (char *[]) { (char *) "admfourone", (char *) "admfourtwo", NULL },
      (char *[]) { NULL } },
    { (char *) "five", (char *) "pwdfive",
      (char *[]) { NULL },
      (char *[]) { (char *) "memfiveone", (char *) "memfivetwo", NULL } },
  };
#define ndata (sizeof (data) / sizeof (data[0]))


static int
do_test (void)
{
  FILE *fp = tmpfile ();
  if (fp == NULL)
    {
      puts ("cannot open temporary file");
      return 1;
    }

  for (size_t i = 0; i < ndata; ++i)
    if (putsgent (&data[i], fp) != 0)
      {
	printf ("putsgent call %zu failed\n", i + 1);
	return 1;
      }

  rewind (fp);

  int result = 0;
  int seen = -1;
  struct sgrp *g;
  while ((g = fgetsgent (fp)) != NULL)
    {
      ++seen;
      if (strcmp (g->sg_namp, data[seen].sg_namp) != 0)
	{
	  printf ("sg_namp of entry %d does not match: %s vs %s\n",
		  seen + 1, g->sg_namp, data[seen].sg_namp);
	  result = 1;
	}
      if (strcmp (g->sg_passwd, data[seen].sg_passwd) != 0)
	{
	  printf ("sg_passwd of entry %d does not match: %s vs %s\n",
		  seen + 1, g->sg_passwd, data[seen].sg_passwd);
	  result = 1;
	}
      if (g->sg_adm == NULL)
	{
	  printf ("sg_adm of entry %d is NULL\n", seen + 1);
	  result = 1;
	}
      else
	{
	  int i = 1;
	  char **sp1 = g->sg_adm;
	  char **sp2 = data[seen].sg_adm;
	  while (*sp1 != NULL && *sp2 != NULL)
	    {
	      if (strcmp (*sp1, *sp2) != 0)
		{
		  printf ("sg_adm[%d] of entry %d does not match: %s vs %s\n",
			  i, seen + 1, *sp1, *sp2);
		  result = 1;
		}
	      ++sp1;
	      ++sp2;
	      ++i;
	    }
	  if (*sp1 == NULL && *sp2 != NULL)
	    {
	      printf ("sg_adm of entry %d has too few entries\n", seen + 1);
	      result = 1;
	    }
	  else if (*sp1 != NULL && *sp2 == NULL)
	    {
	      printf ("sg_adm of entry %d has too many entries\n", seen + 1);
	      result = 1;
	    }
	}
      if (g->sg_mem == NULL)
	{
	  printf ("sg_mem of entry %d is NULL\n", seen + 1);
	  result = 1;
	}
      else
	{
	  int i = 1;
	  char **sp1 = g->sg_mem;
	  char **sp2 = data[seen].sg_mem;
	  while (*sp1 != NULL && *sp2 != NULL)
	    {
	      if (strcmp (*sp1, *sp2) != 0)
		{
		  printf ("sg_mem[%d] of entry %d does not match: %s vs %s\n",
			  i, seen + 1, *sp1, *sp2);
		  result = 1;
		}
	      ++sp1;
	      ++sp2;
	      ++i;
	    }
	  if (*sp1 == NULL && *sp2 != NULL)
	    {
	      printf ("sg_mem of entry %d has too few entries\n", seen + 1);
	      result = 1;
	    }
	  else if (*sp1 != NULL && *sp2 == NULL)
	    {
	      printf ("sg_mem of entry %d has too many entries\n", seen + 1);
	      result = 1;
	    }
	}
    }

  fclose (fp);

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
