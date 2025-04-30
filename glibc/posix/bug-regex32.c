// BZ 12811
#include <regex.h>
#include <stdio.h>
#include <locale.h>

static int
do_test (void)
{
  char buf[1000];
  regex_t preg;
  if (setlocale (LC_CTYPE, "de_DE.UTF-8") == NULL)
    {
      puts ("setlocale failed");
      return 1;
    }

  int e = regcomp (&preg, ".*ab", REG_ICASE);
  if (e != 0)
    {
      regerror (e, &preg, buf, sizeof (buf));
      printf ("regcomp = %d \"%s\"\n", e, buf);
      return 1;
    }

  // Incomplete character at the end of the buffer
  e = regexec (&preg, "aaaaaaaaaaaa\xc4", 0, NULL, 0);

  regfree (&preg);
  regerror (e, &preg, buf, sizeof (buf));
  printf ("regexec = %d \"%s\"\n", e, buf);

  return e != REG_NOMATCH;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
