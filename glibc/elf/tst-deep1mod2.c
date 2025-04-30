#include <stdio.h>
extern int baz (void);
extern int xyzzy (void);
int
bar (void)
{
  printf ("%s:%s\n", __FILE__, __func__);
  return baz () + xyzzy ();;
}

int
back (void)
{
  printf ("%s:%s\n", __FILE__, __func__);
  return -1;
}
