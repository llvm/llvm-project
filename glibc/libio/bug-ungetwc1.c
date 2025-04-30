#define _XOPEN_SOURCE 500
#include <stdio.h>
#include <stdlib.h>
#include <locale.h>
#include <wchar.h>

const char write_chars[] = "ABC";      /* Characters on testfile. */
const wint_t unget_wchar = L'A';      /* Ungotten wide character. */

char *fname;


static int do_test (void);
#define TEST_FUNCTION do_test ()

#include "../test-skeleton.c"


static int
do_test (void)
{
  wint_t wc;
  FILE *fp;
  int fd;

  fname = (char *) malloc (strlen (test_dir) + sizeof "/bug-ungetwc1.XXXXXX");
  if (fname == NULL)
    {
      puts ("no memory");
      return 1;
    }
  strcpy (stpcpy (fname, test_dir), "/bug-ungetwc1.XXXXXX");
  fd = mkstemp (fname);
  if (fd == -1)
    {
      printf ("cannot open temporary file: %m\n");
      return 1;
    }
  add_temp_file (fname);

  setlocale(LC_ALL, "");

  /* Output to the file. */
  if ((fp = fdopen (fd, "w")) == NULL)
    {
      fprintf (stderr, "Cannot make `%s' file\n", fname);
      exit (EXIT_FAILURE);
    }

  fprintf (fp, "%s", write_chars);
  fclose (fp);

  /* Read from the file. */
  fp = fopen (fname, "r");

  size_t i = 0;
  while (!feof (fp))
    {
      wc = getwc (fp);
      if (i >= sizeof (write_chars))
	{
	  printf ("Did not get end-of-file when expected.\n");
	  return 1;
	}
      else if (wc != (write_chars[i] ? write_chars[i] : WEOF))
	{
	  printf ("Unexpected %lu from getwc.\n", (unsigned long int) wc);
	  return 1;
	}
      i++;
    }
  printf ("\nThe end-of-file indicator is set.\n");

  /* Unget a wide character. */
  ungetwc (unget_wchar, fp);
  printf ("< `%lc' is ungotten.\n", unget_wchar);

  /* Check the end-of-file indicator. */
  if (feof (fp))
    {
      printf ("The end-of-file indicator is still set.\n");
      return 1;
    }
  else
    printf ("The end-of-file flag is cleared.\n");

  fflush (stdout);
  fclose (fp);

  return 0;
}
