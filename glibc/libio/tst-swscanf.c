#include <locale.h>
#include <stdio.h>
#include <string.h>
#include <wchar.h>


static int do_test (const char *loc);


int
main (void)
{
  int result;

  result = do_test ("C");
  result |= do_test ("de_DE.ISO-8859-1");
  result |= do_test ("de_DE.UTF-8");
  result |= do_test ("ja_JP.EUC-JP");

  return result;
}


static const struct
{
  const wchar_t *fmt;
  const wchar_t *wfmt;
  const wchar_t *arg;
  int retval;
  const char *res;
  const wchar_t *wres;
  int only_C_locale;
} tests[] =
  {
    { L"%[abc]", L"%l[abc]", L"aabbccddaabb", 1 ,"aabbcc", L"aabbcc", 0 },
    { L"%[^def]", L"%l[^def]", L"aabbccddaabb", 1, "aabbcc", L"aabbcc", 0 },
    { L"%[^abc]", L"%l[^abc]", L"aabbccddaabb", 0, "", L"", 0 },
    { L"%[a-c]", L"%l[a-c]", L"aabbccddaabb", 1, "aabbcc", L"aabbcc", 1 },
    { L"%[^d-f]", L"%l[^d-f]", L"aabbccddaabb", 1, "aabbcc", L"aabbcc", 1 },
    { L"%[^a-c]", L"%l[^a-c]", L"aabbccddaabb", 0, "", L"", 1 },
    { L"%[^a-c]", L"%l[^a-c]", L"bbccddaabb", 0, "", L"", 1 }
  };


static int
do_test (const char *loc)
{
  size_t n;
  int result = 0;

  if (setlocale (LC_ALL, loc) == NULL)
    {
      printf ("cannot set locale \"%s\": %m\n", loc);
      return 1;
    }

  printf ("\nnew locale: \"%s\"\n", loc);

  for (n = 0; n < sizeof (tests) / sizeof (tests[0]); ++n)
    {
      char buf[100];
      wchar_t wbuf[100];

      if (tests[n].only_C_locale && strcmp (loc, "C") != 0)
	continue;

      if (swscanf (tests[n].arg, tests[n].fmt, buf) != tests[n].retval)
	{
	  printf ("swscanf (\"%S\", \"%S\", ...) failed\n",
		  tests[n].arg, tests[n].fmt);
	  result = 1;
	}
      else if (tests[n].retval != 0 && strcmp (buf, tests[n].res) != 0)
	{
	  printf ("swscanf (\"%S\", \"%S\", ...) return \"%s\", expected \"%s\"\n",
		  tests[n].arg, tests[n].fmt, buf, tests[n].res);
	  result = 1;
	}
      else
	printf ("swscanf (\"%S\", \"%S\", ...) OK\n",
		tests[n].arg, tests[n].fmt);

      if (swscanf (tests[n].arg, tests[n].wfmt, wbuf) != tests[n].retval)
	{
	  printf ("swscanf (\"%S\", \"%S\", ...) failed\n",
		  tests[n].arg, tests[n].wfmt);
	  result = 1;
	}
      else if (tests[n].retval != 0 && wcscmp (wbuf, tests[n].wres) != 0)
	{
	  printf ("swscanf (\"%S\", \"%S\", ...) return \"%S\", expected \"%S\"\n",
		  tests[n].arg, tests[n].wfmt, wbuf, tests[n].wres);
	  result = 1;
	}
      else
	printf ("swscanf (\"%S\", \"%S\", ...) OK\n",
		tests[n].arg, tests[n].wfmt);
    }

  return result;
}
