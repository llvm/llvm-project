#include <stdio.h>

int
__attribute__((noinline))
mod3fn1 (int x)
{
  puts ("in mod3fn1");
  return x + 6;
}

int
mod3fn2 (int x)
{
  puts ("in mod3fn2");
  return mod3fn1 (x / 2) * 2;
}
