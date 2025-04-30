/* glibc test for TLS in ld.so.  */
#include <stdio.h>

#include "tls-macros.h"


/* Two 'int' variables in TLS.  */
VAR_INT_DEF(foo);
VAR_INT_DEF(bar);


static int
do_test (void)
{
  int result = 0;
  int *ap, *bp;


  /* Set the variable using the local exec model.  */
  puts ("set bar to 1 (LE)");
  ap = TLS_LE (bar);
  *ap = 1;


  /* Get variables using initial exec model.  */
  fputs ("get sum of foo and bar (IE)", stdout);
  ap = TLS_IE (foo);
  bp = TLS_IE (bar);
  printf (" = %d\n", *ap + *bp);
  result |= *ap + *bp != 1;
  if (*ap != 0)
    {
      printf ("foo = %d\n", *ap);
      result = 1;
    }
  if (*bp != 1)
    {
      printf ("bar = %d\n", *bp);
      result = 1;
    }


  /* Get variables using local dynamic model.  */
  fputs ("get sum of foo and bar (LD)", stdout);
  ap = TLS_LD (foo);
  bp = TLS_LD (bar);
  printf (" = %d\n", *ap + *bp);
  result |= *ap + *bp != 1;
  if (*ap != 0)
    {
      printf ("foo = %d\n", *ap);
      result = 1;
    }
  if (*bp != 1)
    {
      printf ("bar = %d\n", *bp);
      result = 1;
    }


  /* Get variables using generic dynamic model.  */
  fputs ("get sum of foo and bar (GD)", stdout);
  ap = TLS_GD (foo);
  bp = TLS_GD (bar);
  printf (" = %d\n", *ap + *bp);
  result |= *ap + *bp != 1;
  if (*ap != 0)
    {
      printf ("foo = %d\n", *ap);
      result = 1;
    }
  if (*bp != 1)
    {
      printf ("bar = %d\n", *bp);
      result = 1;
    }

  return result;
}


#include <support/test-driver.c>
