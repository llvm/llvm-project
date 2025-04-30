/* Test case by Miloslav Trmač <mitr@volny.cz>.  */
#include <locale.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

int
main (void)
{
  wchar_t wc;

  if (setlocale (LC_CTYPE, "de_DE.UTF-8") == NULL)
    {
      puts ("setlocale failed");
      return 1;
    }

  if (mbtowc (&wc, "\xc3\xa1", MB_CUR_MAX) != 2 || wc != 0xE1)
    {
      puts ("1st mbtowc failed");
      return 1;
    }

  if (mbtowc (&wc, "\xc3\xa1", SIZE_MAX) != 2 || wc != 0xE1)
    {
      puts ("2nd mbtowc failed");
      return 1;
    }

  return 0;
}
