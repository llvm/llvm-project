#include <stdio.h>
#include <stdlib.h>

int
main (void)
{
  double d;
  int c;

  if (scanf ("%lg", &d) != 0)
    {
      printf ("scanf didn't failed\n");
      exit (1);
    }
  c = getchar ();
  if (c != ' ')
    {
      printf ("c is `%c', not ` '\n", c);
      exit (1);
    }

  return 0;
}
