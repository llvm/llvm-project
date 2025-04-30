/* Taken from the Li18nux base test suite.  */

#define _XOPEN_SOURCE 500
#include <errno.h>
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <wchar.h>

#define WIDE_STR_LEN 32

int
main (int argc, char *argv[])
{
  size_t i;
  FILE   *fp;
  wchar_t *ret, wcs[WIDE_STR_LEN];
  int result = 0;
  const char il_str1[] = {0xe3, 0x81, '\0'};
  const char il_str2[] = {'0', '\n', 'A', 'B', 0xe3, 0x81, 'E', '\0'};
  char name1[] = "/tmp/tst-fgetws.out.XXXXXX";
  char name2[] = "/tmp/tst-fgetws.out.XXXXXX";
  int fd;

  puts ("This program runs on de_DE.UTF-8 locale.");
  if (setlocale (LC_ALL, "de_DE.UTF-8") == NULL)
    {
      fprintf (stderr, "Err: Cannot run on the de_DE.UTF-8 locale");
      exit (EXIT_FAILURE);
    }

  /* Make a file `il_str1'. */
  fd = mkstemp (name1);
  if (fd == -1)
    {
      printf ("cannot open temp file: %m\n");
      exit (EXIT_FAILURE);
    }
  if ((fp = fdopen (fd, "w")) == NULL)
    {
      printf ("Can't open %s.\n", argv[1]);
      exit (EXIT_FAILURE);
    }
  fwrite (il_str1, sizeof (char), sizeof (il_str1), fp);
  fclose (fp);

  /* Make a file `il_str2'. */
  fd = mkstemp (name2);
  if (fd == -1)
    {
      printf ("cannot open temp file: %m\n");
      exit (EXIT_FAILURE);
    }
  if ((fp = fdopen (fd, "w")) == NULL)
    {
      fprintf (stderr, "Can't open %s.\n", argv[1]);
      exit (EXIT_FAILURE);
    }
  fwrite (il_str2, sizeof (char), sizeof (il_str2), fp);
  fclose (fp);


  /* Test for il_str1. */
  if ((fp = fopen (name1, "r")) == NULL)
    {
      fprintf (stderr, "Can't open %s.\n", argv[1]);
      exit (EXIT_FAILURE);
    }

  puts ("--");
  puts ("Read a byte sequence which is invalid as a wide character string.");
  puts (" bytes: 0xe3, 0x81, '\\0'");

  errno = 0;
  ret = fgetws (wcs, WIDE_STR_LEN, fp);

  if (ret == NULL)
    {
      puts ("Return Value: NULL");

      if (errno == EILSEQ)
	puts ("errno = EILSEQ");
      else
	{
	  printf ("errno = %d\n", errno);
	  result = 1;
	}
    }
  else
    {
      printf ("Return Value: %p\n", ret);
      for (i = 0; i < wcslen (wcs) + 1; i++)
	printf (" wcs[%zd] = %04x", i, (unsigned int)wcs[i]);
      printf ("\n");
      result = 1;
    }

  /* Test for il_str2. */
  if ((fp = fopen (name2, "r")) == NULL)
    {
      fprintf (stderr, "Can't open %s.\n", argv[1]);
      exit (EXIT_FAILURE);
    }

  puts ("--");
  puts ("Read a byte sequence which is invalid as a wide character string.");
  puts (" bytes: '0', '\\n', 'A', 'B', 0xe3, 0x81, 'c', '\\0'");

  errno = 0;
  ret = fgetws (wcs, WIDE_STR_LEN, fp);

  if (ret == NULL)
    {
      puts ("Return Value: NULL");

      if (errno == EILSEQ)
	puts ("errno = EILSEQ");
      else
	printf ("errno = %d\n", errno);

      result = 1;
    }
  else
    {
      size_t i;

      printf ("Return Value: %p\n", ret);
      for (i = 0; i < wcslen (wcs) + 1; i++)
	printf (" wcs[%zd] = 0x%04x", i, (unsigned int)wcs[i]);
      printf ("\n");

      for (i = 0; il_str2[i] != '\n'; ++i)
	if ((wchar_t) il_str2[i] != wcs[i])
	  {
	    puts ("read string not correct");
	    result = 1;
	    break;
	  }
      if (il_str2[i] == '\n')
	{
	  if (wcs[i] != L'\n')
	    {
	      puts ("newline missing");
	      result = 1;
	    }
	  else if (wcs[i + 1] != L'\0')
	    {
	      puts ("read string not NUL-terminated");
	      result = 1;
	    }
	}
    }

  puts ("\nsecond line");
  errno = 0;
  ret = fgetws (wcs, WIDE_STR_LEN, fp);

  if (ret == NULL)
    {
      puts ("Return Value: NULL");

      if (errno == EILSEQ)
	puts ("errno = EILSEQ");
      else
	{
	  printf ("errno = %d\n", errno);
	  result = 1;
	}
    }
  else
    {
      printf ("Return Value: %p\n", ret);
      for (i = 0; i < wcslen (wcs) + 1; i++)
	printf (" wcs[%zd] = 0x%04x", i, (unsigned int)wcs[i]);
      printf ("\n");
    }

  fclose (fp);

  unlink (name1);
  unlink (name2);

  return result;
}
