#include <assert.h>
#include <errno.h>
#include <stdio.h>

#ifndef CHAR
# define CHAR char
# define L(str) str
# define FPUTS fputs
# define FSCANF fscanf
#endif


static int
do_test (void)
{
  FILE *fp = tmpfile ();
  if (fp == NULL)
    {
      puts ("cannot open file");
      return 1;
    }

  FPUTS (L("7-11"), fp);
  rewind (fp);

  printf("setting errno to EINTR\n");
  errno = EINTR;

  printf("checking sscanf\n");

  int i, j, n;

  i = j = n = 0;
  FSCANF (fp, L(" %i - %i %n"), &i, &j, &n);
  printf ("found %i-%i (length=%i)\n", i, j, n);

  int result = 0;
  if (i != 7)
    {
      printf ("i is %d, expected 7\n", i);
      result = 1;
    }
  if (j != 11)
    {
      printf ("j is %d, expected 11\n", j);
      result = 1;
    }
  if (n != 4)
    {
      printf ("n is %d, expected 4\n", j);
      result = 1;
    }

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
