#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 10000

static void *arr[N];

static int
do_test (void)
{
  for (int i = 0; i < N; ++i)
    {
      size_t size = random () % 16384;

      if ((arr[i] = malloc (size)) == NULL)
	{
	nomem:
	  puts ("not enough memory");
	  return 0;
	}

      memset (arr[i], size, size);
    }

  void *p = malloc (256);
  if (p == NULL)
    goto nomem;
  memset (p, 1, 256);

  puts ("==================================================================");

  for (int i = 0; i < N; ++i)
    if (i % 13 != 0)
      free (arr[i]);

  puts ("==================================================================");

  malloc_trim (0);

  puts ("==================================================================");

  p = malloc (30000);
  if (p == NULL)
    goto nomem;

  memset (p, 2, 30000);

  malloc_trim (0);

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
