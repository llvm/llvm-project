#include <stdio.h>

extern int bar (void);

int
foo (void)
{
  return 42 + bar ();
}

static void
__attribute__ ((destructor))
fini (void)
{
  putchar ('3');
}
