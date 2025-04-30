#include <ctype.h>
#include <locale.h>
#include <stdio.h>
#include <wctype.h>


static int
do_test (void)
{
  const char *loc = "de_DE.ISO-8859-1";
  if (setlocale (LC_ALL, loc) == NULL)
    {
      printf ("cannot set %s locale\n", loc);
      return 1;
    }
  printf ("selected locale %s\n", loc);

  wint_t win = 0xe4;
  wint_t wex = 0xc4;
  wint_t wch = towupper (win);
  if (wch != wex)
    {
      printf ("towupper(%x) = %x, expected %x\n", win, wch, wex);
      return 1;
    }
  wch = toupper (win);
  if (wch != wex)
    {
      printf ("toupper(%x) = %x, expected %x\n", win, wch, wex);
      return 1;
    }

  win = 0x69;
  wex = 0x49;
  wch = towupper (win);
  if (wch != wex)
    {
      printf ("towupper(%x) = %x, expected %x\n", win, wch, wex);
      return 1;
    }
  wch = toupper (win);
  if (wch != wex)
    {
      printf ("toupper(%x) = %x, expected %x\n", win, wch, wex);
      return 1;
    }

  loc = "tr_TR.ISO-8859-9";
  if (setlocale (LC_ALL, loc) == NULL)
    {
      printf ("cannot set %s locale\n", loc);
      return 1;
    }
  printf ("selected locale %s\n", loc);

  win = 0x69;
  wex = 0x130;
  wch = towupper (win);
  if (wch != wex)
    {
      printf ("towupper(%x) = %x, expected %x\n", win, wch, wex);
      return 1;
    }
  wch = toupper (win);
  wex = 0xdd;
  if (wch != wex)
    {
      printf ("toupper(%x) = %x, expected %x\n", win, wch, wex);
      return 1;
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
