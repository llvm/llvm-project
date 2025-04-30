#include <stdio.h>
int
foo (void)
{
  printf ("%s:%s\n", __FILE__, __func__);
  return 1;
}

int
baz (void)
{
  printf ("%s:%s\n", __FILE__, __func__);
  return 20;
}
