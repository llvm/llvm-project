#include <stdio.h>
#include <string.h>
#include <wchar.h>

int
main (int argc, char *argv[])
{
  int n;
  int result = 0;
  char buf1[20];
  wchar_t wbuf2[20];
  char c3;
  wchar_t wc4;
  int d;

  puts ("Test 1");

  n = wscanf (L"%s %S %c%C %d", buf1, wbuf2, &c3, &wc4, &d);

  if (n != 5 || strcmp (buf1, "Hello") != 0 || wcscmp (wbuf2, L"World") != 0
      || c3 != '!' || wc4 != L'!' || d != 42)
    {
      printf ("*** FAILED, n = %d, buf1 = \"%s\", wbuf2 = L\"%S\", c3 = '%c', wc4 = L'%C', d = %d\n",
	      n, buf1, wbuf2, c3, (wint_t) wc4, d);
      result = 1;
    }

  return result;
}
