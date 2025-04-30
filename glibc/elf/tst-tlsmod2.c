#include <stdio.h>

#include "tls-macros.h"


COMMON_INT_DEF(foo);


int
in_dso (int n, int *caller_foop)
{
  int *foop;
  int result = 0;

  puts ("foo");			/* Make sure PLT is used before macros.  */
  asm ("" ::: "memory");

  foop = TLS_GD (foo);

  if (caller_foop != NULL && foop != caller_foop)
    {
      printf ("callers address of foo differs: %p vs %p\n", caller_foop, foop);
      result = 1;
    }
  else if (*foop != n)
    {
      printf ("foo != %d\n", n);
      result = 1;
    }

  *foop = 16;

  return result;
}
