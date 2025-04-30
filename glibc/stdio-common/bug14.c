/* Test case by Al Viro <aviro@redhat.com>.  */
#include <locale.h>
#include <wchar.h>
#include <stdio.h>
#include <stdlib.h>

/* MB_CUR_MAX multibyte ones (6 UTF+0080, in this case) */
static const char string[] = "\
\xc2\x80\xc2\x80\xc2\x80\xc2\x80\xc2\x80\xc2\x80";

int
main (void)
{
  if (setlocale (LC_ALL, "de_DE.UTF-8") == NULL)
    {
      puts ("cannot set locale");
      exit (1);
    }

  wchar_t s[7];
  int n = sscanf (string, "%l[\x80\xc2]", s);
  if (n != 1)
    {
      printf ("return values %d != 1\n", n);
      exit (1);
    }

  return 0;
}
