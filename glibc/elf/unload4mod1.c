#include <stdio.h>

extern int bar (int);

int
foo (int x)
{
  puts ("in foo");
  return bar (x / 2) + 2;
}
