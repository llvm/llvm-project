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
  wint_t ret, wc, ungetone = 0x00E4;	/* 0x00E4 means `a umlaut'. */
  char fname[] = "/tmp/tst-ungetwc1.out.XXXXXX";
  int fd;
  int result = 0;

  puts ("This program runs on de_DE.UTF-8 locale.");
  if (setlocale (LC_ALL, "de_DE.UTF-8") == NULL)
    {
      fprintf (stderr, "Err: Cannot run on the de_DE.UTF-8 locale");
      exit (EXIT_FAILURE);
    }

  fd = mkstemp (fname);
  if (fd == -1)
    {
      printf ("cannot open temp file: %m\n");
      exit (EXIT_FAILURE);
    }

  /* Write some characters to `testfile'. */
  if ((fp = fdopen (fd, "w")) == NULL)
    {
      fprintf (stderr, "Cannot open 'testfile'.");
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

  /* Unget a character. */
  ret = ungetwc (ungetone, fp);
  printf ("Unget a character (0x%04x)\n", (unsigned int) ungetone);
  fflush (stdout);
  if (ret == WEOF)
    {
      puts ("ungetwc() returns NULL.");
      exit (EXIT_SUCCESS);
    }

  /* Reget a character. */
  wc = getwc (fp);
  printf ("Reget a character (0x%04x)\n", (unsigned int) wc);
  fflush (stdout);
  if (wc == ungetone)
    {
      puts ("The ungotten character is equal to the regotten character.");
      fflush (stdout);
    }
  else
    {
      puts ("The ungotten character is not equal to the regotten character.");
      printf ("ungotten one: %04x, regetone: %04x", ungetone, wc);
      fflush (stdout);
      result = 1;
    }
  fclose (fp);

  unlink (fname);

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
