#include <stdio.h>

static void
__attribute__ ((destructor))
fini (void)
{
  putchar ('1');
}
