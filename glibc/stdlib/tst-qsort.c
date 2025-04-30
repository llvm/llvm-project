/* Test case by Paul Eggert <eggert@twinsun.com> */
#include <stdio.h>
#include <stdlib.h>
#include <tst-stack-align.h>

struct big { char c[4 * 1024]; };

struct big *array;
struct big *array_end;

static int align_check;

int
compare (void const *a1, void const *b1)
{
  struct big const *a = a1;
  struct big const *b = b1;

  if (!align_check)
    align_check = TEST_STACK_ALIGN () ? -1 : 1;

  if (! (array <= a && a < array_end
	 && array <= b && b < array_end))
    {
      exit (EXIT_FAILURE);
    }
  return b->c[0] - a->c[0];
}

int
main (int argc, char **argv)
{
  size_t i;
  size_t array_members = argv[1] ? atoi (argv[1]) : 50;
  array = (struct big *) malloc (array_members * sizeof *array);
  if (array == NULL)
    {
      puts ("no memory");
      exit (EXIT_FAILURE);
    }

  array_end = array + array_members;
  for (i = 0; i < array_members; i++)
    array[i].c[0] = i % 128;

  qsort (array, array_members, sizeof *array, compare);

  if (align_check == -1)
    {
      puts ("stack not sufficiently aligned");
      exit (EXIT_FAILURE);
    }

  return 0;
}
