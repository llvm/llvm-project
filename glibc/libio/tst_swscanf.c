#include <stdio.h>
#include <string.h>
#include <wchar.h>

int
main (int argc, char *argv[])
{
  const wchar_t in[] = L"7 + 35 is 42";
  size_t n;
  int a, b, c;
  int result = 0;
  char buf1[20];
  wchar_t wbuf2[20];
  char buf3[20];
  char c4;
  wchar_t wc5;

  puts ("Test 1");
  a = b = c = 0;
  n = swscanf (in, L"%d + %d is %d", &a, &b, &c);
  if (n != 3 || a + b != c || c != 42)
    {
      printf ("*** FAILED, n = %Zu, a = %d, b = %d, c = %d\n", n, a, b, c);
      result = 1;
    }

  puts ("Test 2");
  n = swscanf (L"one two three !!", L"%s %S %s %c%C",
	       buf1, wbuf2, buf3, &c4, &wc5);
  if (n != 5 || strcmp (buf1, "one") != 0 || wcscmp (wbuf2, L"two") != 0
      || strcmp (buf3, "three") != 0 || c4 != '!' || wc5 != L'!')
    {
      printf ("*** FAILED, n = %Zu, buf1 = \"%s\", wbuf2 = L\"%S\", buf3 = \"%s\", c4 = '%c', wc5 = L'%C'\n",
	      n, buf1, wbuf2, buf3, c4, (wint_t) wc5);
      result = 1;
    }

  return result;
}
