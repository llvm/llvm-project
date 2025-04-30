#include <stdio.h>
#include <wchar.h>
#include <sys/types.h>


static wchar_t buf[100];
#define nbuf (sizeof (buf) / sizeof (buf[0]))
static const struct
{
  size_t n;
  const char *str;
  ssize_t exp;
} tests[] =
  {
    { nbuf, "hello world", 11 },
    { 0, "hello world", -1 },
    { 0, "", -1 },
    { nbuf, "", 0 }
  };

int
main (int argc, char *argv[])
{
  size_t n;
  int result = 0;

  puts ("test 1");
  n = swprintf (buf, nbuf, L"Hello %s", "world");
  if (n != 11)
    {
      printf ("incorrect return value: %zd instead of 11\n", n);
      result = 1;
    }
  else if (wcscmp (buf, L"Hello world") != 0)
    {
      printf ("incorrect string: L\"%ls\" instead of L\"Hello world\"\n", buf);
      result = 1;
    }

  puts ("test 2");
  n = swprintf (buf, nbuf, L"Is this >%g< 3.1?", 3.1);
  if (n != 18)
    {
      printf ("incorrect return value: %zd instead of 18\n", n);
      result = 1;
    }
  else if (wcscmp (buf, L"Is this >3.1< 3.1?") != 0)
    {
      printf ("incorrect string: L\"%ls\" instead of L\"Is this >3.1< 3.1?\"\n",
	      buf);
      result = 1;
    }

  for (n = 0; n < sizeof (tests) / sizeof (tests[0]); ++n)
    {
      ssize_t res = swprintf (buf, tests[n].n, L"%s", tests[n].str);

      if (tests[n].exp < 0 && res >= 0)
	{
	  printf ("swprintf (buf, %Zu, L\"%%s\", \"%s\") expected to fail\n",
		  tests[n].n, tests[n].str);
	  result = 1;
	}
      else if (tests[n].exp >= 0 && tests[n].exp != res)
	{
	  printf ("swprintf (buf, %Zu, L\"%%s\", \"%s\") expected to return %Zd, but got %Zd\n",
		  tests[n].n, tests[n].str, tests[n].exp, res);
	  result = 1;
	}
      else
	printf ("swprintf (buf, %Zu, L\"%%s\", \"%s\") OK\n",
		tests[n].n, tests[n].str);
    }

  if (swprintf (buf, nbuf, L"%.0s", "foo") != 0
      || wcslen (buf) != 0)
    {
      printf ("swprintf (buf, %Zu, L\"%%.0s\", \"foo\") create some output\n",
	      nbuf);
      result = 1;
    }

  if (swprintf (buf, nbuf, L"%.0ls", L"foo") != 0
      || wcslen (buf) != 0)
    {
      printf ("swprintf (buf, %Zu, L\"%%.0ls\", L\"foo\") create some output\n",
	      nbuf);
      result = 1;
    }

  return result;
}
