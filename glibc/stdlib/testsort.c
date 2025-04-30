#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static int
compare (const void *a, const void *b)
{
  return strcmp (*(char **) a, *(char **) b);
}

int
main (void)
{
  char bufs[500][20];
  char *lines[500];
  size_t lens[500];
  size_t i, j;

  srandom (1);

  for (i = 0; i < 500; ++i)
    {
      lens[i] = random() % 19;
      lines[i] = bufs[i];
      for (j = 0; j < lens[i]; ++j)
	lines[i][j] = random() % 26 + 'a';
      lines[i][j] = '\0';
    }

  qsort (lines, 500, sizeof (char *), compare);

  for (i = 0; i < 500 && lines[i] != NULL; ++i)
    puts (lines[i]);

  return 0;
}
