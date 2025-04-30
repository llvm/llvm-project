/* This program is to generate one of the examples in stdio.texi.  */

#include <stdio.h>


static void
print (double v)
{
  printf ("|%13.4a|%13.4f|%13.4e|%13.4g|\n", v, v, v, v);
}


int
main (void)
{
  print (0.0);
  print (0.5);
  print (1.0);
  print (-1.0);
  print (100.0);
  print (1000.0);
  print (10000.0);
  print (12345.0);
  print (100000.0);
  print (123456.0);

  return 0;
}
