/* Provide a dummy DSO for tst-rec-dlopen to use.  */
#include <stdio.h>
#include <stdlib.h>

int
dummy1 (void)
{
  printf ("Called dummy1()\n");
  return 1;
}
