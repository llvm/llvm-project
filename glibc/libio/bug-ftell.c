#include <locale.h>
#include <stdio.h>
#include <wchar.h>
#include <sys/types.h>


static int
do_test (void)
{
  if (setlocale (LC_ALL, "de_DE.UTF-8") == NULL)
    {
      puts ("setlocale failed");
      return 1;
    }

  FILE *fp = tmpfile ();
  if (fp == NULL)
    {
      puts ("tmpfile failed");
      return 1;
    }

  if (fputws (L"hello", fp) == EOF)
    {
      puts ("fputws failed");
      return 1;
    }

  rewind (fp);

  const wchar_t *cp;
  unsigned int cnt;
  for (cp = L"hello", cnt = 1; *cp != L'\0'; ++cp, ++cnt)
    {
      wint_t wc = fgetwc (fp);
      if (wc != (wint_t) *cp)
	{
	  printf ("fgetwc failed: got L'%lc', expected L'%lc'\n",
		  wc, (wint_t) *cp);
	  return 1;
	}
      off_t o = ftello (fp);
      if (o != cnt)
	{
	  printf ("ftello failed: got %lu, expected %u\n",
		  (unsigned long int) o, cnt);
	  return 1;
	}
      printf ("round %u OK\n", cnt);
    }

  fclose (fp);

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
