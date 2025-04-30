/* Test STT_GNU_IFUNC symbols in PIE:

   1. Direct function call.
   2. Function pointer.
   3. Reference from a shared library.
 */

#include <stdlib.h>
#include "ifunc-sel.h"

typedef int (*foo_p) (void);

static int
one (void)
{
  return -30;
}

void * foo_ifunc (void) __asm__ ("foo");
__asm__(".type foo, %gnu_indirect_function");

void *
inhibit_stack_protector
foo_ifunc (void)
{
  return ifunc_one (one);
}

extern int foo (void);
extern int call_foo (void);
extern foo_p get_foo_p (void);

foo_p foo_ptr = foo;

int
main (void)
{
  foo_p p;

  if (call_foo () != -30)
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
