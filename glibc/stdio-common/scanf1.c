#include  <stdio.h>
#include  <stdlib.h>

int
main (int argc, char *argv[])
{
  int i,n,r;

  n = i = r = -1;
  r = sscanf ("1234:567", "%d%n", &i, &n);
  printf ("%d %d %d\n", r, n, i);
  if (r != 1 || i != 1234 || n != 4)
    abort ();
  return 0;
}
