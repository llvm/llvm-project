/* Test local STT_GNU_IFUNC symbols:

   1. Direct function call.
   2. Function pointer.
 */

#include <stdlib.h>
#include "ifunc-sel.h"

extern int foo (void);

static int
one (void)
{
  return -30;
}

static void * foo_ifunc (void) __asm__ ("foo");
__asm__(".type foo, %gnu_indirect_function");

static void *
__attribute__ ((used))
inhibit_stack_protector
foo_ifunc (void)
{
  return ifunc_one (one);
}

typedef int (*foo_p) (void);

foo_p foo_ptr = foo;

foo_p
__attribute__ ((noinline))
get_foo_p (void)
{
  return foo_ptr;
}

foo_p
__attribute__ ((noinline))
get_foo (void)
{
  return foo;
}

int
main (void)
{
  foo_p p;

  p = get_foo ();
  if (p != foo)
    abort ();
  if ((*p) () != -30)
    abort ();

  p = get_foo_p ();
  if (p != foo)
    abort ();
  if ((*p) () != -30)
    abort ();

  if (foo_ptr != foo)
    abort ();
  if ((*foo_ptr) () != -30)
    abort ();
  if (foo () != -30)
    abort ();

  return 0;
}
