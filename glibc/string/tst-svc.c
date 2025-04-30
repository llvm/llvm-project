/* Test for strverscmp() */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define  MAX_STRINGS      256
#define  MAX_LINE_SIZE    32

static int
compare (const void *p1, const void *p2)
{
  return strverscmp (*((char **) p1), *((char **) p2));
}

int
do_test (void)
{
  char line[MAX_LINE_SIZE + 1];
  char *str[MAX_STRINGS];
  int  count = 0;
  int  i, n;

  while (count < MAX_STRINGS && fgets (line, MAX_LINE_SIZE, stdin) != NULL)
    {
      n = strlen (line) - 1;

      if (line[n] == '\n')
        line[n] = '\0';

      str[count] = strdup (line);

      if (str[count] == NULL)
        exit (EXIT_FAILURE);

      ++count;
    }

  qsort (str, count, sizeof (char *), compare);

  for (i = 0; i < count; ++i)
    puts (str[i]);

  return EXIT_SUCCESS;
}

#include <support/test-driver.c>
