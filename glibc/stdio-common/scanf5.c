#include <stdio.h>
#include <stdlib.h>

int
main(int argc, char *argv[])
{
  int a, b;

  a = b = -1;
  sscanf ("12ab", "%dab%n", &a, &b);
  printf ("%d, %d\n", a, b);
  if (a != 12 || b != 4)
    abort ();

  a = b = -1;
  sscanf ("12ab100", "%dab%n100", &a, &b);
  printf ("%d, %d\n", a, b);
  if (a != 12 || b != 4)
    abort ();
  return 0;
}
