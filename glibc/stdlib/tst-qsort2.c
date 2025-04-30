#include <stdio.h>
#include <stdlib.h>

char *array;
char *array_end;
size_t member_size;

int
compare (const void *a1, const void *b1)
{
  const char *a = a1;
  const char *b = b1;

  if (! (array <= a && a < array_end
	 && array <= b && b < array_end))
    {
      puts ("compare arguments not inside of the array");
      exit (EXIT_FAILURE);
    }
  int ret = b[0] - a[0];
  if (ret)
    return ret;
  if (member_size > 1)
    return b[1] - a[1];
  return 0;
}

int
test (size_t nmemb, size_t size)
{
  array = malloc (nmemb * size);
  if (array == NULL)
    {
      printf ("%zd x %zd: no memory", nmemb, size);
      return 1;
    }

  array_end = array + nmemb * size;
  member_size = size;

  char *p;
  size_t i;
  size_t bias = random ();
  for (i = 0, p = array; i < nmemb; i++, p += size)
    {
      p[0] = (char) (i + bias);
      if (size > 1)
	p[1] = (char) ((i + bias) >> 8);
    }

  qsort (array, nmemb, size, compare);

  for (i = 0, p = array; i < nmemb - 1; i++, p += size)
    {
      if (p[0] < p[size]
	  || (size > 1 && p[0] == p[size] && p[1] < p[size + 1]))
	{
	  printf ("%zd x %zd: failure at offset %zd\n", nmemb,
		  size, i);
	  free (array);
	  return 1;
	}
    }

  free (array);
  return 0;
}

int
main (int argc, char **argv)
{
  int ret = 0;
  if (argc >= 3)
    ret |= test (atoi (argv[1]), atoi (argv[2]));
  else
    {
      ret |= test (10000, 1);
      ret |= test (200000, 2);
      ret |= test (2000000, 3);
      ret |= test (2132310, 4);
      ret |= test (1202730, 7);
      ret |= test (1184710, 8);
      ret |= test (272710, 12);
      ret |= test (14170, 32);
      ret |= test (4170, 320);
    }

  return ret;
}
