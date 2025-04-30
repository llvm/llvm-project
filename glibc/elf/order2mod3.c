#include <stdio.h>

int
bar (void)
{
  return 1;
}

static void
__attribute__ ((destructor))
fini (void)
{
  putchar ('4');
}
