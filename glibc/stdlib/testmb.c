#include <stdio.h>
#include <stdlib.h>

int
main (int argc, char *argv[])
{
  wchar_t w[10];
  char c[10];
  int i;
  int lose = 0;

  i = mbstowcs (w, "bar", 4);
  if (!(i == 3 && w[1] == 'a'))
    {
      puts ("mbstowcs FAILED!");
      lose = 1;
    }

  mbstowcs (w, "blah", 5);
  i = wcstombs (c, w, 10);
  if (i != 4)
    {
      puts ("wcstombs FAILED!");
      lose = 1;
    }

  if (mblen ("foobar", 7) != 1)
    {
      puts ("mblen 1 FAILED!");
      lose = 1;
    }

  if (mblen ("", 1) != 0)
    {
      puts ("mblen 2 FAILED!");
      lose = 1;
    }

  {
    int r;
    char c = 'x';
    wchar_t wc;
    char mbc[MB_CUR_MAX];

    if ((r = mbtowc (&wc, &c, MB_CUR_MAX)) <= 0)
      {
	printf ("conversion to wide failed, result: %d\n", r);
	lose = 1;
      }
    else
      {
	printf ("wide value: 0x%04lx\n", (unsigned long) wc);
	mbc[0] = '\0';
	if ((r = wctomb (mbc, wc)) <= 0)
	  {
	    printf ("conversion to multibyte failed, result: %d\n", r);
	    lose = 1;
	  }
      }

  }

  puts (lose ? "Test FAILED!" : "Test succeeded.");
  return lose;
}
