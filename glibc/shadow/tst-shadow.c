#include <shadow.h>
#include <stdio.h>
#include <string.h>


static const struct spwd data[] =
  {
    { (char *) "one", (char *) "pwdone", 1, 2, 3, 4, 5, 6, 7 },
    { (char *) "two", (char *) "pwdtwo", 11, 12, 13, 14, 15, 16, 17 },
    { (char *) "three", (char *) "pwdthree", -1, 22, 23, 24, 25, 26, 27 },
    { (char *) "four", (char *) "pwdfour", 31, -1, 33, 34, 35, 36, 37 },
    { (char *) "five", (char *) "pwdfive", 41, 42, -1, 44, 45, 46, 47 },
    { (char *) "six", (char *) "pwdsix", 51, 52, 53, -1, 55, 56, 57 },
    { (char *) "seven", (char *) "pwdseven", 61, 62, 63, 64, -1, 66, 67 },
    { (char *) "eight", (char *) "pwdeigth", 71, 72, 73, 74, 75, -1, 77 },
    { (char *) "nine", (char *) "pwdnine", 81, 82, 83, 84, 85, 86, ~0ul },
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
    if (putspent (&data[i], fp) != 0)
      {
	printf ("putspent call %zu failed\n", i + 1);
	return 1;
      }

  rewind (fp);

  int result = 0;
  int seen = -1;
  struct spwd *p;
  while ((p = fgetspent (fp)) != NULL)
    {
      ++seen;
      if (strcmp (p->sp_namp, data[seen].sp_namp) != 0)
	{
	  printf ("sp_namp of entry %d does not match: %s vs %s\n",
		  seen + 1, p->sp_namp, data[seen].sp_namp);
	  result = 1;
	}
      if (strcmp (p->sp_pwdp, data[seen].sp_pwdp) != 0)
	{
	  printf ("sp_pwdp of entry %d does not match: %s vs %s\n",
		  seen + 1, p->sp_pwdp, data[seen].sp_pwdp);
	  result = 1;
	}
#define T(f) \
      if (p->f != data[seen].f)						      \
	{								      \
	  printf ("%s of entry %d wrong: %ld vs %ld\n",			      \
		  #f, seen + 1, p->f, data[seen].f);			      \
	  result = 1;							      \
	}
      T (sp_lstchg);
      T (sp_min);
      T (sp_max);
      T (sp_warn);
      T (sp_expire);
      if (p->sp_flag != data[seen].sp_flag)
	{
	  printf ("sp_flag of entry %d wrong: %lu vs %lu\n",
		  seen + 1, p->sp_flag, data[seen].sp_flag);
	  result = 1;
	}
    }

  fclose (fp);

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
