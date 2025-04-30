#include <mcheck.h>
#include <stdio.h>
#include <stdlib.h>


#ifndef CHAR_T
# define CHAR_T char
# define W(o) o
# define OPEN_MEMSTREAM open_memstream
#endif

#define S(s) S1 (s)
#define S1(s) #s


static void
mcheck_abort (enum mcheck_status ev)
{
  printf ("mecheck failed with status %d\n", (int) ev);
  exit (1);
}


static int
do_test (void)
{
  mcheck_pedantic (mcheck_abort);

  CHAR_T *buf = (CHAR_T *) 1l;
  size_t len = 12345;
  FILE *fp = OPEN_MEMSTREAM (&buf, &len);
  if (fp == NULL)
    {
      printf ("%s failed\n", S(OPEN_MEMSTREAM));
      return 1;
    }

  if (fflush (fp) != 0)
    {
      puts ("fflush failed");
      return 1;
    }

  if (len != 0)
    {
      puts ("string after no write not empty");
      return 1;
    }
  if (buf == (CHAR_T *) 1l)
    {
      puts ("buf not updated");
      return 1;
    }
  if (buf[0] != W('\0'))
    {
      puts ("buf[0] != 0");
      return 1;
    }

  buf = (CHAR_T *) 1l;
  len = 12345;
  if (fclose (fp) != 0)
    {
      puts ("fclose failed");
      return 1;
    }

  if (len != 0)
    {
      puts ("string after close with no write not empty");
      return 1;
    }
  if (buf == (CHAR_T *) 1l)
    {
      puts ("buf not updated");
      return 1;
    }
  if (buf[0] != W('\0'))
    {
      puts ("buf[0] != 0");
      return 1;
    }

  free (buf);

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
