#include <stdio.h>
#include <string.h>

static int
do_test (void)
{
  FILE *f = tmpfile ();
  char obuf[99999], ibuf[sizeof obuf];
  char *line;
  size_t linesz;

  if (! f)
    {
      perror ("tmpfile");
      return 1;
    }

  if (fputs ("line\n", f) == EOF)
    {
      perror ("fputs");
      return 1;
    }

  memset (obuf, 'z', sizeof obuf);
  memset (ibuf, 'y', sizeof ibuf);

  if (fwrite (obuf, sizeof obuf, 1, f) != 1)
    {
      perror ("fwrite");
      return 1;
    }

  rewind (f);

  line = NULL;
  linesz = 0;
  if (getline (&line, &linesz, f) != 5)
    {
      perror ("getline");
      return 1;
    }
  if (strcmp (line, "line\n"))
    {
      puts ("Lines differ.  Test FAILED!");
      return 1;
    }

  if (fread (ibuf, sizeof ibuf, 1, f) != 1)
    {
      perror ("fread");
      return 1;
    }

  if (memcmp (ibuf, obuf, sizeof ibuf))
    {
      puts ("Buffers differ.  Test FAILED!");
      return 1;
    }

  asprintf (&line, "\
GDB is free software and you are welcome to distribute copies of it\n\
 under certain conditions; type \"show copying\" to see the conditions.\n\
There is absolutely no warranty for GDB; type \"show warranty\" for details.\n\
");

  puts ("Test succeeded.");
  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
