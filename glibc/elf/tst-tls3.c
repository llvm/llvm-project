/* glibc test for TLS in ld.so.  */
#include <stdio.h>

#include "tls-macros.h"


/* One define int variable, two externs.  */
COMMON_INT_DECL(foo);
VAR_INT_DECL(bar);
VAR_INT_DEF(baz);


extern int in_dso (void);


static int
do_test (void)
{
  int result = 0;
  int *ap, *bp, *cp;


  /* Set the variable using the local exec model.  */
  puts ("set baz to 3 (LE)");
  ap = TLS_LE (baz);
  *ap = 3;


  /* Get variables using initial exec model.  */
  puts ("set variables foo and bar (IE)");
  ap = TLS_IE (foo);
  *ap = 1;
  bp = TLS_IE (bar);
  *bp = 2;


  /* Get variables using local dynamic model.  */
  fputs ("get sum of foo, bar (GD) and baz (LD)", stdout);
  ap = TLS_GD (foo);
  bp = TLS_GD (bar);
  cp = TLS_LD (baz);
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


  result |= in_dso ();

  return result;
}


#include <support/test-driver.c>
