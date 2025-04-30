/* Based on a test program by Won Kyu Park <wkpark@chem.skku.ac.kr>.  */

#include <wchar.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wctype.h>
#include <locale.h>

int
main (void)
{
  int test = 0;
  int idx = 0;
  char buf[100], *pchar;
  wchar_t tmp[10];
  wchar_t tmp1[] = { L'W', L'o', L'r', L'l', L'd', L'\0' };
  char str[] = "Hello";
  int result = 0;

  pchar = setlocale (LC_ALL, "de_DE.UTF-8");
  printf ("locale : %s\n",pchar);
  printf ("MB_CUR_MAX %Zd\n", MB_CUR_MAX);

  puts ("---- test 1 ------");
  test = mbstowcs (tmp, str, (strlen (str) + 1) * sizeof (char));
  printf ("size of string by mbstowcs %d\n", test);
  if (test != strlen (str))
    result = 1;
  idx += wctomb (&buf[0], tmp[0]);
  idx += wctomb (&buf[idx], tmp[1]);
  buf[idx] = 0;
  printf ("orig string %s\n", str);
  printf ("string by wctomb %s\n", buf);
  printf ("string by %%C %C", (wint_t) tmp[0]);
  if (tmp[0] != L'H')
    result = 1;
  printf ("%C\n", (wint_t) tmp[1]);
  if (tmp[1] != L'e')
    result = 1;
  printf ("string by %%S %S\n", tmp);
  if (wcscmp (tmp, L"Hello") != 0)
    result = 1;
  puts ("---- test 2 ------");
  printf ("wchar string %S\n", tmp1);
  printf ("wchar %C\n", (wint_t) tmp1[0]);
  test = wcstombs (buf, tmp1, (wcslen (tmp1) + 1) * sizeof (wchar_t));
  printf ("size of string by wcstombs %d\n", test);
  if (test != wcslen (tmp1))
    result = 1;
  test = wcslen (tmp1);
  printf ("size of string by wcslen %d\n", test);
  printf ("char %s\n", buf);
  if (strcmp (buf, "World") != 0)
    result = 1;
  puts ("------------------");

  return result;
}
