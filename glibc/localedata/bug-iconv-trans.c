#include <iconv.h>
#include <locale.h>
#include <stdio.h>
#include <string.h>

int
main (void)
{
  iconv_t cd;
  const char str[] = "ƒд÷ц№ья";
  const char expected[] = "AEaeOEoeUEuess";
  char *inptr = (char *) str;
  size_t inlen = strlen (str) + 1;
  char outbuf[500];
  char *outptr = outbuf;
  size_t outlen = sizeof (outbuf);
  int result = 0;
  size_t n;

  if (setlocale (LC_ALL, "de_DE.UTF-8") == NULL)
    {
      puts ("setlocale failed");
      return 1;
    }

  cd = iconv_open ("ANSI_X3.4-1968//TRANSLIT", "ISO-8859-1");
  if (cd == (iconv_t) -1)
    {
      puts ("iconv_open failed");
      return 1;
    }

  n = iconv (cd, &inptr, &inlen, &outptr, &outlen);
  if (n != 7)
    {
      if (n == (size_t) -1)
	printf ("iconv() returned error: %m\n");
      else
	printf ("iconv() returned %Zd, expected 7\n", n);
      result = 1;
    }
  if (inlen != 0)
    {
      puts ("not all input consumed");
      result = 1;
    }
  else if (inptr - str != strlen (str) + 1)
    {
      printf ("inptr wrong, advanced by %td\n", inptr - str);
      result = 1;
    }
  if (memcmp (outbuf, expected, sizeof (expected)) != 0)
    {
      printf ("result wrong: \"%.*s\", expected: \"%s\"\n",
	      (int) (sizeof (outbuf) - outlen), outbuf, expected);
      result = 1;
    }
  else if (outlen != sizeof (outbuf) - sizeof (expected))
    {
      printf ("outlen wrong: %Zd, expected %Zd\n", outlen,
	      sizeof (outbuf) - 15);
      result = 1;
    }
  else
    printf ("output is \"%s\" which is OK\n", outbuf);

  return result;
}
