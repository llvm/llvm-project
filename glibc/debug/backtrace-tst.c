#include <execinfo.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>


static int
compare (const void *p1, const void *p2)
{
  void *ba[20];
  int n = backtrace (ba, sizeof (ba) / sizeof (ba[0]));
  if (n != 0)
    {
      char **names = backtrace_symbols (ba, n);
      if (names != NULL)
	{
	  int i;
	  printf ("called from %s\n", names[0]);
	  for (i = 1; i < n; ++i)
	    printf ("            %s\n", names[i]);
	  free (names);
	}
    }

  return *(const uint32_t *) p1 - *(const uint32_t *) p2;
}


int
main (int argc, char *argv[])
{
  uint32_t arr[20];
  size_t cnt;

  for (cnt = 0; cnt < sizeof (arr) / sizeof (arr[0]); ++cnt)
    arr[cnt] = random ();

  qsort (arr, sizeof (arr) / sizeof (arr[0]), sizeof (arr[0]), compare);

  for (cnt = 0; cnt < sizeof (arr) / sizeof (arr[0]); ++cnt)
    printf ("%" PRIx32 "\n", arr[cnt]);

  return 0;
}
