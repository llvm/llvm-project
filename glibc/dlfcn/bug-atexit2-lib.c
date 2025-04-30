#include <stdio.h>
#include <stdlib.h>

void
fx (void)
{
  puts ("At exit fx");
}

void
foo (void)
{
  atexit (fx);
}
