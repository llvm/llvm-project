#include <time.h>
#include <wchar.h>

int
main (int argc, char *argv[])
{
  wchar_t buf[200];
  time_t t;
  struct tm *tp;
  int result = 0;
  size_t n;

  t = time (NULL);
  tp = gmtime (&t);

  n = wcsftime (buf, sizeof (buf) / sizeof (buf[0]),
		L"%H:%M:%S  %Y-%m-%d\n", tp);
  if (n != 21)
    result = 1;

  wprintf (L"It is now %ls", buf);

  wcsftime (buf, sizeof (buf) / sizeof (buf[0]), L"%A\n", tp);

  wprintf (L"The weekday is %ls", buf);

  return result;
}
