#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


static char fname[] = "/tmp/rndseek.XXXXXX";
static char tempdata[65 * 1024];


static int do_test (void);
#define TEST_FUNCTION do_test ()

#include "../test-skeleton.c"


static int
fp_test (const char *name, FILE *fp)
{
  int result = 0;
  int rounds = 10000;

  do
    {
      int idx = random () % (sizeof (tempdata) - 2);
      char ch1;
      char ch2;

      if (fseek (fp, idx, SEEK_SET) != 0)
	{
	  printf ("%s: %d: fseek failed: %m\n", name, rounds);
	  result = 1;
	  break;
	}

      ch1 = fgetc_unlocked (fp);
      ch2 = tempdata[idx];
      if (ch1 != ch2)
	{
	  printf ("%s: %d: character at index %d not what is expected ('%c' vs '%c')\n",
		  name, rounds, idx, ch1, ch2);
	  result = 1;
	  break;
	}

      ch1 = fgetc (fp);
      ch2 = tempdata[idx + 1];
      if (ch1 != ch2)
	{
	  printf ("%s: %d: character at index %d not what is expected ('%c' vs '%c')\n",
		  name, rounds, idx + 1, ch1, ch2);
	  result = 1;
	  break;
	}
    }
  while (--rounds > 0);

  fclose (fp);

  return result;
}


static int
do_test (void)
{
  int fd;
  FILE *fp;
  size_t i;
  int result;

  fd = mkstemp (fname);
  if (fd == -1)
    {
      printf ("cannot open temporary file: %m\n");
      return 1;
    }
  /* Make sure the file gets removed.  */
  add_temp_file (fname);

  /* Repeatability demands this.  */
  srandom (42);

  /* First create some temporary data.  */
  for (i = 0; i < sizeof (tempdata); ++i)
    tempdata[i] = 'a' + random () % 26;

  /* Write this data to a file.  */
  if (TEMP_FAILURE_RETRY (write (fd, tempdata, sizeof (tempdata)))
      != sizeof (tempdata))
    {
      printf ("cannot wrote data to temporary file: %m\n");
      return 1;
    }

  /* Now try reading the data.  */
  fp = fdopen (dup (fd), "r");
  if (fp == NULL)
    {
      printf ("cannot duplicate temporary file descriptor: %m\n");
      return 1;
    }

  rewind (fp);
  for (i = 0; i < sizeof (tempdata); ++i)
    {
      int ch0 = fgetc (fp);
      char ch1 = ch0;
      char ch2 = tempdata[i];

      if (ch0 == EOF)
	{
	  puts ("premature end of file while reading data");
	  return 1;
	}

      if (ch1 != ch2)
	{
	  printf ("%zd: '%c' vs '%c'\n", i, ch1, ch2);
	  return 1;
	}
    }

  result = fp_test ("fdopen(\"r\")", fp);

  fp = fopen (fname, "r");
  result |= fp_test ("fopen(\"r\")", fp);

  fp = fopen64 (fname, "r");
  result |= fp_test ("fopen64(\"r\")", fp);

  /* The "rw" mode will prevent the mmap-using code from being used.  */
  fp = fdopen (fd, "rw");
  result = fp_test ("fdopen(\"rw\")", fp);

  fp = fopen (fname, "rw");
  result |= fp_test ("fopen(\"rw\")", fp);

  fp = fopen64 (fname, "rw");
  result |= fp_test ("fopen64(\"rw\")", fp);

  return result;
}
