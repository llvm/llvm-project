#include <stdio.h>
#include <stdlib.h>

int
__attribute__((noinline))
baz (int x)
{
  abort ();
}

int
bar (int x)
{
  puts ("in bar");
  return baz (x + 1) + 2;
}
