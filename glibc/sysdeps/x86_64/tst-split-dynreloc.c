/* This test will be used to create an executable with a specific
   section layout in which .rela.dyn and .rela.plt are not contiguous.
   For x86 case, readelf will report something like:

   ...
   [10] .rela.dyn         RELA
   [11] .bar              PROGBITS
   [12] .rela.plt         RELA
   ...

   This is important as this case was not correctly handled by dynamic
   linker in the bind-now case, and the second section was never
   processed.  */

#include <stdio.h>

const int __attribute__ ((section(".bar"))) bar = 0x12345678;
static const char foo[] = "foo";

static int
do_test (void)
{
  printf ("%s %d\n", foo, bar);
  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
