/* Test STT_GNU_IFUNC symbols with dynamic function pointer only.  */

#include <stdlib.h>

extern int foo (void);
extern int foo_protected (void);

typedef int (*foo_p) (void);

foo_p
__attribute__ ((noinline))
get_foo (void)
{
  return foo;
}

foo_p
__attribute__ ((noinline))
get_foo_protected (void)
{
  return foo_protected;
}

int
main (void)
{
  foo_p p;

  p = get_foo ();
  if ((*p) () != -1)
    abort ();

  p = get_foo_protected ();
  if ((*p) () != 0)
    abort ();

  return 0;
}
