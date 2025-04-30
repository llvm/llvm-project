#include <stdio.h>

extern int foo (void);
extern int bar (void);

void
__attribute__ ((constructor))
init (void)
{
  (void) (foo () - bar ());
}

static void
__attribute__ ((destructor))
fini (void)
{
  putchar ('2');
}
