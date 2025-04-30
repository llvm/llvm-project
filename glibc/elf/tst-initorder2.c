#include <stdio.h>

#ifndef NAME
int
main (void)
{
  puts ("main");
}
#else
static void __attribute__ ((constructor))
init (void)
{
  puts ("init: " NAME);
}
static void __attribute__ ((destructor))
fini (void)
{
  puts ("fini: " NAME);
}
#endif
