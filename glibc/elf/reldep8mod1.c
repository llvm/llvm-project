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
baz (void)
{
}
