#include <stdio.h>

#include "tls-macros.h"


/* One define int variable, two externs.  */
COMMON_INT_DEF(foo);
VAR_INT_DEF(bar);
VAR_INT_DECL(baz);

extern int in_dso (void);

int
in_dso (void)
{
  int result = 0;
  int *ap, *bp, *cp;

  /* Get variables using initial exec model.  */
  fputs ("get sum of foo and bar (IE)", stdout);
  asm ("" ::: "memory");
  ap = TLS_IE (foo);
  bp = TLS_IE (bar);
  printf (" = %d\n", *ap + *bp);
  result |= *ap + *bp != 3;
  if (*ap != 1)
    {
      printf ("foo = %d\n", *ap);
      result = 1;
    }
  if (*bp != 2)
    {
      printf ("bar = %d\n", *bp);
      result = 1;
    }


  /* Get variables using generic dynamic model.  */
  fputs ("get sum of foo and bar and baz (GD)", stdout);
  ap = TLS_GD (foo);
  bp = TLS_GD (bar);
  cp = TLS_GD (baz);
  printf (" = %d\n", *ap + *bp + *cp);
  result |= *ap + *bp + *cp != 6;
  if (*ap != 1)
    {
      printf ("foo = %d\n", *ap);
      result = 1;
    }
  if (*bp != 2)
    {
      printf ("bar = %d\n", *bp);
      result = 1;
    }
  if (*cp != 3)
    {
      printf ("baz = %d\n", *cp);
      result = 1;
    }

  return result;
}
