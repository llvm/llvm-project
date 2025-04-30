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

  for (int outer = 0; outer < 800; ++outer)
    {
      for (int inner = 0; inner < 100; ++inner)
	if (fputc (W('a') + (outer * 100 + inner) % 26, fp) == EOF)
	  {
	    printf ("fputc at %d:%d failed\n", outer, inner);
	    return 1;
	  }

      if (fflush (fp) != 0)
	{
	  puts ("fflush failed");
	  return 1;
	}

      if (len != (outer + 1) * 100)
	{
	  printf ("string in round %d not %d bytest long\n",
		  outer + 1, (outer + 1) * 100);
	  return 1;
	}
      if (buf == (CHAR_T *) 1l)
	{
	  printf ("round %d: buf not updated\n", outer + 1);
	  return 1;
	}
      for (int inner = 0; inner < (outer + 1) * 100; ++inner)
	if (buf[inner] != W('a') + inner % 26)
	  {
	    printf ("round %d: buf[%d] != '%c'\n", outer + 1, inner,
		    (char) (W('a') + inner % 26));
	    return 1;
	  }
    }

  buf = (CHAR_T *) 1l;
  len = 12345;
  if (fclose (fp) != 0)
    {
      puts ("fclose failed");
      return 1;
    }

  if (len != 800 * 100)
    {
      puts ("string after close not 80000 bytes long");
      return 1;
    }
  if (buf == (CHAR_T *) 1l)
    {
      puts ("buf not updated");
      return 1;
    }
  for (int inner = 0; inner < 800 * 100; ++inner)
    if (buf[inner] != W('a') + inner % 26)
      {
	printf ("after close: buf[%d] != %c\n", inner,
		(char) (W('a') + inner % 26));
	return 1;
      }

  free (buf);

  return 0;
}

#define TIMEOUT 100
#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
