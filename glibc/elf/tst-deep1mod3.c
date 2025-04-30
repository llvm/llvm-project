#include <stdio.h>

extern int back (void);

int
baz (void)
{
  printf ("%s:%s\n", __FILE__, __func__);
  return back ();
}

int
xyzzy (void)
{
  printf ("%s:%s\n", __FILE__, __func__);
  return 0;
}
