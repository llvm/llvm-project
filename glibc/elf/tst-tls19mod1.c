#include <stdio.h>

extern int bar (void);
extern int baz (void);

int
foo (void)
{
  int v1 = bar ();
  int v2 = baz ();

  printf ("bar=%d, baz=%d\n", v1, v2);

  return v1 != 666 || v2 != 42;
}
