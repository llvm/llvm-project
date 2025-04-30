#include <locale.h>
#include <stdio.h>
#include <wchar.h>


static int
do_test (void)
{
  if (setlocale (LC_ALL, "de_DE.UTF-8") == NULL)
    {
      puts ("setlocale failed");
      return 1;
    }

  if (setvbuf (stdin, NULL, _IONBF, 0) != 0)
    {
      puts ("setvbuf failed");
      return 1;
    }

  wchar_t buf[100];
  size_t nbuf = 0;
  wint_t c;
  while ((c = fgetwc (stdin)) != WEOF)
    buf[nbuf++] = c;

  if (ferror (stdin))
    {
      puts ("error on stdin");
      return 1;
    }

  const wchar_t expected[] =
    {
      0x00000439, 0x00000446, 0x00000443, 0x0000043a,
      0x00000435, 0x0000043d, 0x0000000a, 0x00000071,
      0x00000077, 0x00000065, 0x00000072, 0x00000074,
      0x00000079, 0x0000000a
    };

  if (nbuf != sizeof (expected) / sizeof (expected[0])
      || wmemcmp (expected, buf, nbuf) != 0)
    {
      puts ("incorrect result");
      return 1;
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
