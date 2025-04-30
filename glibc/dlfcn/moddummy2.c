/* Provide a dummy DSO for tst-rec-dlopen to use.  */
#include <stdio.h>
#include <stdlib.h>

int
dummy2 (void)
{
  printf ("Called dummy2()\n");
  /* If the outer dlopen is not dummy1 (becuase of some error)
     then tst-rec-dlopen will see a value of -1 as the returned
     result and fail.  */
  return -1;
}
