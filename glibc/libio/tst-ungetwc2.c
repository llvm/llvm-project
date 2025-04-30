/* Taken from the Li18nux base test suite.  */

#define _XOPEN_SOURCE 500
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <wchar.h>

static int
do_test (void)
{
  FILE *fp;
  const char *str = "abcdef";
  wint_t ret, wc;
  char fname[] = "/tmp/tst-ungetwc2.out.XXXXXX";
  int fd;
  long int pos;
  int result = 0;

  puts ("This program runs on de_DE.UTF-8 locale.");
  if (setlocale (LC_ALL, "de_DE.UTF-8") == NULL)
    {
      fprintf (stderr, "Err: Cannot run on the de_DE.UTF-8 locale\n");
      exit (EXIT_FAILURE);
    }

  /* Write some characters to `testfile'. */
  fd = mkstemp (fname);
  if (fd == -1)
    {
      printf ("cannot open temp file: %m\n");
      exit (EXIT_FAILURE);
    }
  if ((fp = fdopen (fd, "w")) == NULL)
    {
      fprintf (stderr, "Cannot open 'testfile'.\n");
      exit (EXIT_FAILURE);
    }
  fputs (str, fp);
  fclose (fp);

  /* Open `testfile'. */
  if ((fp = fopen (fname, "r")) == NULL)
    {
      fprintf (stderr, "Cannot open 'testfile'.");
      exit (EXIT_FAILURE);
    }

  /* Get a character. */
  wc = getwc (fp);
  pos = ftell (fp);
  printf ("After get a character: %ld\n", pos);
  if (pos != 1)
    result = 1;

  /* Unget a character. */
  ret = ungetwc (wc, fp);
  if (ret == WEOF)
    {
      fprintf (stderr, "ungetwc() returns NULL.");
      exit (EXIT_FAILURE);
    }
  pos = ftell (fp);
  printf ("After unget a character: %ld\n", pos);
  if (pos != 0)
    result = 1;

  /* Reget a character. */
  wc = getwc (fp);
  pos = ftell (fp);
  printf ("After reget a character: %ld\n", pos);
  if (pos != 1)
    result = 1;

  fclose (fp);

  unlink (fname);

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
