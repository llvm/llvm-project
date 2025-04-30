#include <stdlib.h>
void
foo (void)
{
  exit (0);
}

void
__attribute__((destructor))
bar (void)
{
  static int i;
  foo ();
  ++i;
}

void
__attribute__((constructor))
destr (void)
{
  extern void baz (void);
  baz ();
}
